/**
 * ProcessEngine.cpp
 *
 * Implements the asynchronous QProcess-based subprocess controller for
 * SeedVR2 inference_cli.py.  Stdout is read char-by-char (line-buffered)
 * via the Qt event loop so the UI thread never blocks even during 24-hour
 * batch renders.
 */

#include "ProcessEngine.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QTimer>
#include <QtGlobal>

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

ProcessEngine::ProcessEngine(QObject *parent)
    : QObject(parent)
    , m_process(new QProcess(this))
    , m_stopping(false)
{
    m_process->setProcessChannelMode(QProcess::MergedChannels);

    connect(m_process, &QProcess::readyRead,
            this, &ProcessEngine::onReadyRead);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &ProcessEngine::onProcessFinished);
    connect(m_process, &QProcess::errorOccurred,
            this, &ProcessEngine::onErrorOccurred);
}

ProcessEngine::~ProcessEngine()
{
    stopProcess();
}

// ─────────────────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────────────────

void ProcessEngine::startProcess(const QString &pythonExe,
                                 const QString &cliScript,
                                 const QStringList &args,
                                 const QProcessEnvironment &extraEnv)
{
    if (isRunning()) {
        emit logLine(QStringLiteral("⚠  A process is already running. Stop it first.\n"));
        return;
    }

    m_stopping   = false;
    m_lineBuffer.clear();

    // ── Build environment ──────────────────────────────────────────────────
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    // Force UTF-8 output from the Python child process on Windows
    env.insert(QStringLiteral("PYTHONIOENCODING"),              QStringLiteral("utf-8"));
    env.insert(QStringLiteral("PYTHONLEGACYWINDOWSFSENCODING"), QStringLiteral("1"));

    // Suppress expandable_segments allocator warnings via env as an extra
    // safety net (inference_cli.py also sets this via warnings.filterwarnings)
    env.insert(QStringLiteral("PYTHONWARNINGS"), QStringLiteral("ignore::UserWarning"));
    env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    env.insert(QStringLiteral("SEEDVR2_STRICT_MEMORY_FLUSH"), QStringLiteral("1"));

    // Blackwell / CUDA performance variables
    env.insert(QStringLiteral("PYTORCH_ALLOC_CONF"),
               QStringLiteral("backend:cudaMallocAsync,max_split_size_mb:256,"
                               "garbage_collection_threshold:0.6"));
    env.insert(QStringLiteral("TORCH_CUDNN_BENCHMARK"),       QStringLiteral("1"));
    env.insert(QStringLiteral("CUDA_MODULE_LOADING"),         QStringLiteral("LAZY"));
    env.insert(QStringLiteral("TORCH_CUDNN_V8_API_ENABLED"),  QStringLiteral("1"));
    env.insert(QStringLiteral("PYTORCH_NO_CUDA_MEMORY_CACHING"), QStringLiteral("0"));
    env.insert(QStringLiteral("CUDA_CACHE_MAXSIZE"),          QStringLiteral("4294967296"));
    env.insert(QStringLiteral("NVIDIA_TF32_OVERRIDE"),        QStringLiteral("1"));
    env.insert(QStringLiteral("ATTENTION_BACKEND"),           QStringLiteral("sageattention"));

    // Merge caller-supplied overrides
    for (const QString &key : extraEnv.keys()) {
        env.insert(key, extraEnv.value(key));
    }

    m_process->setProcessEnvironment(env);

    // ── Launch ────────────────────────────────────────────────────────────
    QStringList fullArgs;
    fullArgs << cliScript << args;

    emit logLine(QStringLiteral("▶  ") + pythonExe + QStringLiteral(" ") + fullArgs.join(QLatin1Char(' ')) + QStringLiteral("\n"));

    m_process->start(pythonExe, fullArgs);
    if (!m_process->waitForStarted(5000)) {
        emit logLine(QStringLiteral("❌  Failed to start process: ") + m_process->errorString() + QStringLiteral("\n"));
        emit processingFinished(false, QStringLiteral("Failed to start process."));
    }
}

void ProcessEngine::stopProcess()
{
    if (!isRunning())
        return;

    m_stopping = true;
    m_process->terminate();

    // Give the process 8 s to exit cleanly, then force-kill the tree.
    QTimer::singleShot(8000, this, [this]() {
        if (isRunning()) {
            m_process->kill();
        }
    });
}

bool ProcessEngine::isRunning() const
{
    return m_process->state() != QProcess::NotRunning;
}

// ─────────────────────────────────────────────────────────────────────────────
// Private slots
// ─────────────────────────────────────────────────────────────────────────────

void ProcessEngine::onReadyRead()
{
    // Accumulate raw bytes and emit complete lines; handles partial reads.
    m_lineBuffer += m_process->readAll();

    int nl = -1;
    while ((nl = m_lineBuffer.indexOf('\n')) != -1) {
        QByteArray raw = m_lineBuffer.left(nl);
        m_lineBuffer.remove(0, nl + 1);

        // Strip carriage-return for Windows line endings
        if (!raw.isEmpty() && raw.back() == '\r')
            raw.chop(1);

        parseLine(QString::fromUtf8(raw));
    }
}

void ProcessEngine::onProcessFinished(int exitCode, QProcess::ExitStatus /*status*/)
{
    // Flush any remaining buffered output
    if (!m_lineBuffer.isEmpty()) {
        parseLine(QString::fromUtf8(m_lineBuffer));
        m_lineBuffer.clear();
    }

    if (m_stopping) {
        emit logLine(QStringLiteral("\n⏹  Processing cancelled by user.\n"));
        emit processingFinished(false, QStringLiteral("Cancelled."));
    } else if (exitCode == 0) {
        emit logLine(QStringLiteral("\n✅  Processing completed successfully.\n"));
        emit processingFinished(true, QStringLiteral("Done."));
    } else {
        emit logLine(QStringLiteral("\n❌  Process exited with code ") + QString::number(exitCode) + QStringLiteral(".\n"));
        emit processingFinished(false, QStringLiteral("Exit code ") + QString::number(exitCode) + QStringLiteral("."));
    }
}

void ProcessEngine::onErrorOccurred(QProcess::ProcessError error)
{
    if (error == QProcess::FailedToStart) {
        emit logLine(QStringLiteral("❌  Python executable not found. Check the path in settings.\n"));
        emit processingFinished(false, QStringLiteral("Python executable not found."));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Line parsing
// ─────────────────────────────────────────────────────────────────────────────

void ProcessEngine::parseLine(const QString &line)
{
    if (tryParseStatusToken(line))
        return;   // structured status token – do not echo to log

    emit logLine(line);
    tryParseProgressToken(line);
}

bool ProcessEngine::tryParseStatusToken(const QString &line)
{
    const QLatin1String prefix(k_statusPrefix);
    if (!line.startsWith(prefix))
        return false;

    const QString jsonStr = line.mid(static_cast<int>(qstrlen(k_statusPrefix)));
    const QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
    if (doc.isNull() || !doc.isObject())
        return false;

    const QJsonObject obj = doc.object();
    const QString filename  = obj.value(QLatin1String("file_path")).toString();
    const int currentFrame  = obj.value(QLatin1String("current")).toInt(0);
    const int totalFrames   = obj.value(QLatin1String("total")).toInt(0);
    const int doneFiles     = obj.value(QLatin1String("done")).toInt(0);
    const int remainingFiles = obj.value(QLatin1String("remaining")).toInt(0);
    const int remainingFramesQueue = obj.value(QLatin1String("remaining_frames_queue")).toInt(0);

    emit fileProgressUpdated(filename, currentFrame, totalFrames, doneFiles, remainingFiles, remainingFramesQueue);
    return true;
}

void ProcessEngine::tryParseProgressToken(const QString &line)
{
    // Fallback: parse "step N/M" or "steps N/M" tokens for inner batch bar.
    const QString lower = line.toLower();
    const QStringList stepTokens  = { QStringLiteral("step "), QStringLiteral("steps: "), QStringLiteral("steps ") };

    for (const QString &token : stepTokens) {
        int idx = lower.indexOf(token);
        if (idx == -1)
            continue;

        int start = idx + token.length();
        int slashPos = lower.indexOf(QLatin1Char('/'), start);
        if (slashPos == -1)
            continue;

        // Extract the number before and after '/'
        QString curStr;
        for (int i = start; i < slashPos; ++i) {
            if (lower[i].isDigit()) curStr += lower[i];
        }
        QString totStr;
        for (int i = slashPos + 1; i < lower.length() && lower[i].isDigit(); ++i) {
            totStr += lower[i];
        }

        if (curStr.isEmpty() || totStr.isEmpty())
            continue;

        bool okC = false, okT = false;
        int c = curStr.toInt(&okC);
        int t = totStr.toInt(&okT);
        if (okC && okT && t > 0) {
            emit batchProgressUpdated(c, t);
            return;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

QString ProcessEngine::buildFormatSeconds(double seconds)
{
    if (seconds < 0.0) seconds = 0.0;
    const int total = static_cast<int>(seconds);
    const int hh    = total / 3600;
    const int mm    = (total % 3600) / 60;
    const int ss    = total % 60;
    return QStringLiteral("%1:%2:%3")
        .arg(hh, 2, 10, QLatin1Char('0'))
        .arg(mm, 2, 10, QLatin1Char('0'))
        .arg(ss, 2, 10, QLatin1Char('0'));
}
