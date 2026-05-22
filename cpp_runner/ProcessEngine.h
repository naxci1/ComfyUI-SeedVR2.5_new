/**
 * ProcessEngine.h
 *
 * Asynchronous QProcess-based subprocess controller for SeedVR2 inference_cli.py.
 * Parses __SEEDVR2_GUI_STATUS__ JSON tokens and fallback step/batch/frame tokens
 * from stdout in real-time without blocking the Qt event loop.
 */

#pragma once

#include <QObject>
#include <QProcess>
#include <QStringList>
#include <QByteArray>

class ProcessEngine : public QObject
{
    Q_OBJECT

public:
    explicit ProcessEngine(QObject *parent = nullptr);
    ~ProcessEngine() override;

    /**
     * Launch inference_cli.py as a subprocess.
     *
     * @param pythonExe   Full path to the Python executable.
     * @param cliScript   Full path to inference_cli.py.
     * @param args        CLI arguments forwarded verbatim to inference_cli.py.
     * @param extraEnv    Optional additional environment variable overrides.
     */
    void startProcess(const QString &pythonExe,
                      const QString &cliScript,
                      const QStringList &args,
                      const QProcessEnvironment &extraEnv = QProcessEnvironment());

    /** Request a graceful stop followed by a force-kill if the process does not exit. */
    void stopProcess();

    /** Returns true if the subprocess is currently running. */
    bool isRunning() const;

signals:
    /** Emitted for every decoded log line (UTF-8). */
    void logLine(const QString &line);

    /**
     * Emitted when a __SEEDVR2_GUI_STATUS__ token is decoded.
     *
     * @param filename      Currently processed file path.
     * @param currentFrame  Frame index within the current file.
     * @param totalFrames   Total frames in the current file.
     * @param doneFiles     Number of files fully processed.
     * @param remainingFiles Number of files still queued (excluding current).
     * @param remainingFramesQueue Total frames still pending in upcoming queue files.
     */
    void fileProgressUpdated(const QString &filename,
                             int currentFrame,
                             int totalFrames,
                             int doneFiles,
                             int remainingFiles,
                             int remainingFramesQueue);

    /** Emitted when a step/batch token is parsed (inner progress bar). */
    void batchProgressUpdated(int current, int total);

    /** Emitted when the subprocess exits. */
    void processingFinished(bool success, const QString &message);

private slots:
    void onReadyRead();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onErrorOccurred(QProcess::ProcessError error);

private:
    void parseLine(const QString &line);
    bool tryParseStatusToken(const QString &line);
    void tryParseProgressToken(const QString &line);
    static QString buildFormatSeconds(double seconds);

    QProcess   *m_process;
    QByteArray  m_lineBuffer;   // incomplete UTF-8 line accumulator
    bool        m_stopping;

    static constexpr const char *k_statusPrefix = "__SEEDVR2_GUI_STATUS__|";
};
