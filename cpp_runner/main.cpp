#include "ProcessEngine.h"

#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QDockWidget>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBox>
#include <QVBoxLayout>
#include <QComboBox>
#include <QCheckBox>
#include <QIcon>
#include <QDateTime>
#include <QUrl>
#include <QtGlobal>

namespace {

constexpr const char *kStyleSheet = R"(
QMainWindow, QWidget {
    background: #181818;
    color: #f0f0f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QFrame#panel, QFrame#previewPane, QFrame#timelinePane {
    background: #1f1f1f;
    border: 1px solid #333;
    border-radius: 8px;
}
QFrame#sidebar {
    background: #252525;
    border-left: 1px solid #3a3a3a;
}
QLabel#previewLabel {
    background: #101010;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #9f9f9f;
}
QLineEdit, QComboBox, QSpinBox {
    background: #202020;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 4px 6px;
}
QPushButton {
    background: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover { background: #373737; }
QPushButton#startBtn { background: #1565c0; border: none; color: white; }
QPushButton#stopBtn { background: #8e2d2d; border: none; color: white; }
QProgressBar {
    background: #111;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    min-height: 18px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2d8cf0;
    border-radius: 6px;
}
QToolBox::tab {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px;
    font-weight: 600;
}
QToolBox::tab:selected { background: #343434; }
QDockWidget::title {
    background: #252525;
    text-align: left;
    padding-left: 8px;
    border: 1px solid #3a3a3a;
}
QPlainTextEdit {
    background: #0f0f0f;
    border: none;
    color: #d0d0d0;
    font-family: "JetBrains Mono", "Consolas", monospace;
}
)";

QString uniquePath(const QString &path)
{
    QFileInfo fi(path);
    if (!fi.exists()) return fi.absoluteFilePath();

    const QString dir = fi.absolutePath();
    const QString base = fi.completeBaseName();
    const QString ext = fi.suffix();

    int n = 1;
    while (true) {
        const QString candidate = ext.isEmpty()
            ? QStringLiteral("%1/%2_%3").arg(dir, base).arg(n)
            : QStringLiteral("%1/%2_%3.%4").arg(dir, base).arg(n).arg(ext);
        if (!QFileInfo::exists(candidate)) return candidate;
        ++n;
    }
}

bool isImagePath(const QString &path)
{
    const QString ext = QFileInfo(path).suffix().toLower();
    return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" ||
           ext == "webp" || ext == "tif" || ext == "tiff";
}

QStringList ffmpegExtractArgs(const QString &input, double sec, const QString &outPng)
{
    return {
        QStringLiteral("-y"),
        QStringLiteral("-ss"), QString::number(qMax(0.0, sec), 'f', 3),
        QStringLiteral("-i"), input,
        QStringLiteral("-frames:v"), QStringLiteral("1"),
        QStringLiteral("-q:v"), QStringLiteral("2"),
        outPng
    };
}

QWidget *pathRow(QWidget *parent, QLineEdit *edit, QPushButton *browse)
{
    auto *w = new QWidget(parent);
    auto *l = new QHBoxLayout(w);
    l->setContentsMargins(0, 0, 0, 0);
    l->setSpacing(6);
    l->addWidget(edit, 1);
    l->addWidget(browse);
    return w;
}

} // namespace

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow()
        : QMainWindow(nullptr)
        , m_engine(new ProcessEngine(this))
    {
        setWindowTitle(QStringLiteral("SeedVR2 Video AI"));
        setWindowIcon(QIcon(QStringLiteral("app.ico")));
        setMinimumSize(1680, 980);
        setAcceptDrops(true);

        if (qApp) qApp->setStyleSheet(QString::fromLatin1(kStyleSheet));

        buildUi();
        wireSignals();
        loadConfigJsonOnStartup();

        statusBar()->showMessage(QStringLiteral("Ready"));
        appendLog(QStringLiteral("[INFO] UI initialized."));
    }

    ~MainWindow() override = default;

protected:
    void closeEvent(QCloseEvent *event) override
    {
        if (m_engine->isRunning()) m_engine->stopProcess();
        QMainWindow::closeEvent(event);
    }

    void dragEnterEvent(QDragEnterEvent *event) override
    {
        if (event->mimeData()->hasUrls()) event->acceptProposedAction();
    }

    void dropEvent(QDropEvent *event) override
    {
        const QList<QUrl> urls = event->mimeData()->urls();
        if (urls.isEmpty()) return;

        const QString dropped = urls.first().toLocalFile();
        if (dropped.isEmpty()) return;

        m_inputPathEdit->setText(dropped);
        m_currentInputPath = dropped;
        appendLog(QStringLiteral("[INFO] Dropped input: %1").arg(dropped));

        loadFirstFrame(dropped);
        buildTimelineThumbnails(dropped);
    }

private slots:
    void appendEngineLog(const QString &line)
    {
        appendLog(line);
    }

    void onRunClicked()
    {
        const QString cliPath = resolveCliScriptAbsolutePath();
        if (cliPath.isEmpty()) {
            appendLog(QStringLiteral("[ERROR] inference_cli.py could not be resolved from SeedVR2 Folder."));
            QMessageBox::warning(this,
                                 QStringLiteral("CLI Not Found"),
                                 QStringLiteral("Could not locate inference_cli.py.\n"
                                                "Check config.json key 'SeedVR2 Folder' or the sidebar path."));
            return;
        }

        QString pythonExe = m_pythonExeEdit->text().trimmed();
        if (pythonExe.isEmpty()) pythonExe = QStringLiteral("python");

        const QString inputPath = m_inputPathEdit->text().trimmed();
        if (inputPath.isEmpty() || !QFileInfo::exists(inputPath)) {
            appendLog(QStringLiteral("[ERROR] Input path missing or invalid."));
            return;
        }

        const QString outDir = m_outputDirEdit->text().trimmed().isEmpty()
            ? QFileInfo(inputPath).absolutePath()
            : m_outputDirEdit->text().trimmed();

        const QString container = m_containerCombo->currentText().toLower();
        const QString baseOut = QDir(outDir).absoluteFilePath(
            QStringLiteral("%1_upscaled.%2").arg(QFileInfo(inputPath).completeBaseName(), container));
        const QString safeOut = uniquePath(baseOut);

        QStringList args;
        args << inputPath;
        args << QStringLiteral("--output") << QFileInfo(safeOut).absoluteFilePath();
        args << QStringLiteral("--output_format") << container;
        args << QStringLiteral("--video_backend") << m_backendCombo->currentText().toLower();
        args << QStringLiteral("--batch_size") << QString::number(m_batchSizeSpin->value());
        args << QStringLiteral("--resolution") << QString::number(m_resolutionSpin->value());
        args << QStringLiteral("--input_noise_scale") << QString::number(m_inputNoiseSlider->value() / 100.0, 'f', 2);
        args << QStringLiteral("--latent_noise_scale") << QString::number(m_latentNoiseSlider->value() / 100.0, 'f', 2);

        if (!m_lutEdit->text().trimmed().isEmpty()) {
            args << QStringLiteral("--lut") << m_lutEdit->text().trimmed();
        }

        const QJsonArray ffmpegArgs = buildFfmpegArgs();
        if (!ffmpegArgs.isEmpty()) {
            args << QStringLiteral("--ffmpeg_video_args")
                 << QString::fromUtf8(QJsonDocument(ffmpegArgs).toJson(QJsonDocument::Compact));
        }

        if (m_10bitCheck->isChecked()) {
            args << QStringLiteral("--10bit");
        }

        m_targetOutputPath = safeOut;
        appendLog(QStringLiteral("[INFO] Launching render -> %1").arg(safeOut));
        m_engine->startProcess(pythonExe, cliPath, args);
    }

    void onPreviewCurrentFrameClicked()
    {
        if (m_currentInputPath.isEmpty() || !QFileInfo::exists(m_currentInputPath)) {
            appendLog(QStringLiteral("[WARN] No input loaded for preview render."));
            return;
        }

        const QString cliPath = resolveCliScriptAbsolutePath();
        if (cliPath.isEmpty()) {
            appendLog(QStringLiteral("[ERROR] Cannot run preview render: CLI path unresolved."));
            return;
        }

        QString pythonExe = m_pythonExeEdit->text().trimmed();
        if (pythonExe.isEmpty()) pythonExe = QStringLiteral("python");

        // Extract current timeline frame to temporary PNG, then run batch_size=1 preview.
        const double sec = m_durationSec > 0.0
            ? (m_durationSec * m_timelineSlider->value() / 1000.0)
            : 0.0;
        const QString tempFrame = QDir::temp().absoluteFilePath(
            QStringLiteral("seedvr2_preview_input_%1.png").arg(QDateTime::currentMSecsSinceEpoch()));

        if (!extractFrameAt(m_currentInputPath, sec, tempFrame)) {
            appendLog(QStringLiteral("[ERROR] Failed to extract source frame for preview."));
            return;
        }

        const QString previewOut = uniquePath(
            QDir::temp().absoluteFilePath(
                QStringLiteral("seedvr2_preview_output_%1.png").arg(QDateTime::currentMSecsSinceEpoch())));

        QStringList args;
        args << tempFrame;
        args << QStringLiteral("--output") << previewOut;
        args << QStringLiteral("--output_format") << QStringLiteral("png");
        args << QStringLiteral("--batch_size") << QStringLiteral("1");
        args << QStringLiteral("--resolution") << QString::number(m_resolutionSpin->value());

        appendLog(QStringLiteral("[INFO] Running preview render (batch_size=1) ..."));

        auto *previewProc = new QProcess(this);
        connect(previewProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
                [this, previewProc, previewOut](int code, QProcess::ExitStatus) {
            const QString out = QString::fromUtf8(previewProc->readAllStandardOutput());
            const QString err = QString::fromUtf8(previewProc->readAllStandardError());
            if (!out.trimmed().isEmpty()) appendLog(out.trimmed());
            if (!err.trimmed().isEmpty()) appendLog(err.trimmed());

            if (code == 0 && QFileInfo::exists(previewOut)) {
                setPreviewPixmap(m_outputPreview, previewOut);
                appendLog(QStringLiteral("[INFO] Preview frame rendered: %1").arg(previewOut));
            } else {
                appendLog(QStringLiteral("[ERROR] Preview render failed."));
            }
            previewProc->deleteLater();
        });

        previewProc->start(pythonExe, QStringList{cliPath} + args);
    }

    void onStopClicked()
    {
        m_engine->stopProcess();
        appendLog(QStringLiteral("[INFO] Stop requested."));
    }

    void onTimelineScrubbed(int value)
    {
        Q_UNUSED(value)
        if (m_currentInputPath.isEmpty() || isImagePath(m_currentInputPath)) return;
        if (m_durationSec <= 0.0) return;

        const double sec = m_durationSec * m_timelineSlider->value() / 1000.0;
        const QString framePath = QDir::temp().absoluteFilePath(
            QStringLiteral("seedvr2_scrub_%1.png").arg(QDateTime::currentMSecsSinceEpoch()));

        if (extractFrameAt(m_currentInputPath, sec, framePath)) {
            setPreviewPixmap(m_inputPreview, framePath);
        }
    }

    void onFileProgress(const QString &, int currentFrame, int totalFrames, int doneFiles, int remainingFiles, int)
    {
        const int filePct = totalFrames > 0 ? qBound(0, (currentFrame * 100) / totalFrames, 100) : 0;
        m_fileProgress->setValue(filePct);

        const int totalFiles = doneFiles + remainingFiles + 1;
        const int batchPct = totalFiles > 0 ? qBound(0, (doneFiles * 100) / totalFiles, 100) : 0;
        m_batchProgress->setValue(batchPct);
    }

    void onBatchProgress(int current, int total)
    {
        if (total > 0) {
            m_batchProgress->setValue(qBound(0, (current * 100) / total, 100));
        }
    }

    void onProcessingFinished(bool success, const QString &message)
    {
        appendLog(QStringLiteral("[INFO] Process finished: %1").arg(message));
        statusBar()->showMessage(success ? QStringLiteral("Render completed") : QStringLiteral("Render stopped"), 5000);

        if (success && !m_targetOutputPath.isEmpty() && QFileInfo::exists(m_targetOutputPath)) {
            if (isImagePath(m_targetOutputPath)) {
                setPreviewPixmap(m_outputPreview, m_targetOutputPath);
            } else {
                const QString outFrame = QDir::temp().absoluteFilePath(
                    QStringLiteral("seedvr2_output_preview_%1.png").arg(QDateTime::currentMSecsSinceEpoch()));
                if (extractFrameAt(m_targetOutputPath, 0.0, outFrame)) {
                    setPreviewPixmap(m_outputPreview, outFrame);
                }
            }
        }
    }

private:
    void buildUi()
    {
        auto *central = new QWidget(this);
        setCentralWidget(central);

        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(8, 8, 8, 8);
        root->setSpacing(8);

        auto *mainSplit = new QSplitter(Qt::Horizontal, central);
        root->addWidget(mainSplit, 1);

        // Left main area
        auto *left = new QWidget(mainSplit);
        auto *leftLayout = new QVBoxLayout(left);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        leftLayout->setSpacing(8);

        auto *previewPanel = new QFrame(left);
        previewPanel->setObjectName(QStringLiteral("panel"));
        auto *previewPanelLayout = new QVBoxLayout(previewPanel);

        auto *previewSplit = new QSplitter(Qt::Horizontal, previewPanel);
        auto *inPane = new QFrame(previewSplit);
        auto *outPane = new QFrame(previewSplit);
        inPane->setObjectName(QStringLiteral("previewPane"));
        outPane->setObjectName(QStringLiteral("previewPane"));

        auto *inLayout = new QVBoxLayout(inPane);
        auto *outLayout = new QVBoxLayout(outPane);

        auto *inTitle = new QLabel(QStringLiteral("Input"), inPane);
        auto *outTitle = new QLabel(QStringLiteral("Output / Processed"), outPane);

        m_inputPreview = new QLabel(QStringLiteral("Drop video/image to load first frame"), inPane);
        m_outputPreview = new QLabel(QStringLiteral("Output preview"), outPane);
        for (QLabel *lbl : {m_inputPreview, m_outputPreview}) {
            lbl->setObjectName(QStringLiteral("previewLabel"));
            lbl->setAlignment(Qt::AlignCenter);
            lbl->setMinimumSize(540, 360);
        }

        inLayout->addWidget(inTitle);
        inLayout->addWidget(m_inputPreview, 1);
        outLayout->addWidget(outTitle);
        outLayout->addWidget(m_outputPreview, 1);

        previewPanelLayout->addWidget(previewSplit, 1);
        leftLayout->addWidget(previewPanel, 5);

        // Timeline area
        auto *timelinePanel = new QFrame(left);
        timelinePanel->setObjectName(QStringLiteral("timelinePane"));
        auto *timelineLayout = new QVBoxLayout(timelinePanel);

        auto *timelineLabel = new QLabel(QStringLiteral("Timeline"), timelinePanel);
        m_timelineSlider = new QSlider(Qt::Horizontal, timelinePanel);
        m_timelineSlider->setRange(0, 1000);
        m_timelineSlider->setValue(0);

        auto *thumbScroll = new QScrollArea(timelinePanel);
        thumbScroll->setWidgetResizable(true);
        thumbScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        thumbScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        thumbScroll->setFixedHeight(130);

        m_thumbHost = new QWidget(thumbScroll);
        m_thumbLayout = new QHBoxLayout(m_thumbHost);
        m_thumbLayout->setContentsMargins(6, 6, 6, 6);
        m_thumbLayout->setSpacing(8);
        thumbScroll->setWidget(m_thumbHost);

        timelineLayout->addWidget(timelineLabel);
        timelineLayout->addWidget(m_timelineSlider);
        timelineLayout->addWidget(thumbScroll);

        leftLayout->addWidget(timelinePanel, 2);

        // Control row
        auto *controls = new QHBoxLayout();
        m_previewBtn = new QPushButton(QStringLiteral("Preview Current Frame"), left);
        m_runBtn = new QPushButton(QStringLiteral("Start Export"), left);
        m_runBtn->setObjectName(QStringLiteral("startBtn"));
        m_stopBtn = new QPushButton(QStringLiteral("Stop"), left);
        m_stopBtn->setObjectName(QStringLiteral("stopBtn"));

        controls->addWidget(m_previewBtn);
        controls->addStretch(1);
        controls->addWidget(m_runBtn);
        controls->addWidget(m_stopBtn);
        leftLayout->addLayout(controls);

        m_fileProgress = new QProgressBar(left);
        m_batchProgress = new QProgressBar(left);
        leftLayout->addWidget(new QLabel(QStringLiteral("File Progress"), left));
        leftLayout->addWidget(m_fileProgress);
        leftLayout->addWidget(new QLabel(QStringLiteral("Batch Progress"), left));
        leftLayout->addWidget(m_batchProgress);

        // Right sidebar
        auto *sidebar = new QFrame(mainSplit);
        sidebar->setObjectName(QStringLiteral("sidebar"));
        auto *sidebarLayout = new QVBoxLayout(sidebar);

        auto *sidebarScroll = new QScrollArea(sidebar);
        sidebarScroll->setWidgetResizable(true);
        auto *sidebarContent = new QWidget(sidebarScroll);
        auto *sidebarContentLayout = new QVBoxLayout(sidebarContent);

        m_toolbox = new QToolBox(sidebarContent);

        QWidget *pathsPage = new QWidget(m_toolbox);
        auto *pathsForm = new QFormLayout(pathsPage);
        m_inputPathEdit = new QLineEdit(pathsPage);
        m_outputDirEdit = new QLineEdit(pathsPage);
        m_seedFolderEdit = new QLineEdit(pathsPage);
        m_pythonExeEdit = new QLineEdit(pathsPage);
        m_lutEdit = new QLineEdit(pathsPage);

        auto *inputBrowse = new QPushButton(QStringLiteral("Browse"), pathsPage);
        auto *outputBrowse = new QPushButton(QStringLiteral("Browse"), pathsPage);
        auto *seedBrowse = new QPushButton(QStringLiteral("Browse"), pathsPage);

        pathsForm->addRow(QStringLiteral("Input:"), pathRow(pathsPage, m_inputPathEdit, inputBrowse));
        pathsForm->addRow(QStringLiteral("Output Directory:"), pathRow(pathsPage, m_outputDirEdit, outputBrowse));
        pathsForm->addRow(QStringLiteral("SeedVR2 Folder:"), pathRow(pathsPage, m_seedFolderEdit, seedBrowse));
        pathsForm->addRow(QStringLiteral("Python Executable:"), m_pythonExeEdit);
        pathsForm->addRow(QStringLiteral("LUT:"), m_lutEdit);

        QWidget *adjustPage = new QWidget(m_toolbox);
        auto *adjustForm = new QFormLayout(adjustPage);
        m_resolutionSpin = new QSpinBox(adjustPage);
        m_resolutionSpin->setRange(128, 7680);
        m_resolutionSpin->setValue(1080);
        m_batchSizeSpin = new QSpinBox(adjustPage);
        m_batchSizeSpin->setRange(1, 200);
        m_batchSizeSpin->setValue(81);

        m_inputNoiseSlider = new QSlider(Qt::Horizontal, adjustPage);
        m_inputNoiseSlider->setRange(0, 100);
        m_inputNoiseSlider->setValue(0);

        m_latentNoiseSlider = new QSlider(Qt::Horizontal, adjustPage);
        m_latentNoiseSlider->setRange(0, 100);
        m_latentNoiseSlider->setValue(0);

        adjustForm->addRow(QStringLiteral("Resolution:"), m_resolutionSpin);
        adjustForm->addRow(QStringLiteral("Batch Size:"), m_batchSizeSpin);
        adjustForm->addRow(QStringLiteral("Input Noise:"), m_inputNoiseSlider);
        adjustForm->addRow(QStringLiteral("Latent Noise:"), m_latentNoiseSlider);

        QWidget *codecPage = new QWidget(m_toolbox);
        auto *codecForm = new QFormLayout(codecPage);

        m_containerCombo = new QComboBox(codecPage);
        m_containerCombo->addItems({QStringLiteral("mp4"), QStringLiteral("mkv"), QStringLiteral("mov"), QStringLiteral("webm"), QStringLiteral("avi")});

        m_codecCombo = new QComboBox(codecPage);
        m_codecCombo->addItems({QStringLiteral("H265"), QStringLiteral("H264"), QStringLiteral("AV1"), QStringLiteral("VP9")});

        m_backendCombo = new QComboBox(codecPage);
        m_backendCombo->addItems({QStringLiteral("ffmpeg"), QStringLiteral("opencv")});

        m_bitrateModeCombo = new QComboBox(codecPage);
        m_bitrateModeCombo->addItems({QStringLiteral("Dynamic"), QStringLiteral("Constant")});

        m_constantBitrateSpin = new QSpinBox(codecPage);
        m_constantBitrateSpin->setRange(1, 500);
        m_constantBitrateSpin->setValue(80);
        m_constantBitrateSpin->setSuffix(QStringLiteral(" Mbps"));

        m_crfSpin = new QSpinBox(codecPage);
        m_crfSpin->setRange(0, 51);
        m_crfSpin->setValue(12);

        m_presetCombo = new QComboBox(codecPage);
        m_presetCombo->addItems({QStringLiteral("ultrafast"), QStringLiteral("fast"), QStringLiteral("medium"), QStringLiteral("slow")});
        m_presetCombo->setCurrentText(QStringLiteral("medium"));

        m_10bitCheck = new QCheckBox(QStringLiteral("Enable 10-bit"), codecPage);
        m_10bitCheck->setChecked(true);

        codecForm->addRow(QStringLiteral("Container:"), m_containerCombo);
        codecForm->addRow(QStringLiteral("Codec:"), m_codecCombo);
        codecForm->addRow(QStringLiteral("Backend:"), m_backendCombo);
        codecForm->addRow(QStringLiteral("Bitrate Mode:"), m_bitrateModeCombo);
        codecForm->addRow(QStringLiteral("Constant Bitrate:"), m_constantBitrateSpin);
        codecForm->addRow(QStringLiteral("Dynamic CRF/QP:"), m_crfSpin);
        codecForm->addRow(QStringLiteral("Preset:"), m_presetCombo);
        codecForm->addRow(QString(), m_10bitCheck);

        m_toolbox->addItem(pathsPage, QStringLiteral("Paths"));
        m_toolbox->addItem(adjustPage, QStringLiteral("Adjustments"));
        m_toolbox->addItem(codecPage, QStringLiteral("Codec Settings"));

        sidebarContentLayout->addWidget(m_toolbox);
        sidebarContentLayout->addStretch(1);
        sidebarScroll->setWidget(sidebarContent);

        sidebarLayout->addWidget(sidebarScroll, 1);

        mainSplit->addWidget(left);
        mainSplit->addWidget(sidebar);
        mainSplit->setStretchFactor(0, 7);
        mainSplit->setStretchFactor(1, 3);

        // Docked log console
        m_logDock = new QDockWidget(QStringLiteral("Log Console"), this);
        m_logDock->setAllowedAreas(Qt::BottomDockWidgetArea);
        m_logDock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

        m_logConsole = new QPlainTextEdit(m_logDock);
        m_logConsole->setReadOnly(true);
        m_logConsole->setMaximumBlockCount(20000);
        m_logDock->setWidget(m_logConsole);

        addDockWidget(Qt::BottomDockWidgetArea, m_logDock);

        connect(inputBrowse, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Select Input"), m_inputPathEdit->text(),
                                                              QStringLiteral("Media (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
            if (!path.isEmpty()) {
                m_inputPathEdit->setText(path);
                m_currentInputPath = path;
                loadFirstFrame(path);
                buildTimelineThumbnails(path);
            }
        });

        connect(outputBrowse, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), m_outputDirEdit->text());
            if (!path.isEmpty()) m_outputDirEdit->setText(path);
        });

        connect(seedBrowse, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select SeedVR2 Folder"), m_seedFolderEdit->text());
            if (!path.isEmpty()) m_seedFolderEdit->setText(path);
        });
    }

    void wireSignals()
    {
        connect(m_runBtn, &QPushButton::clicked, this, &MainWindow::onRunClicked);
        connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::onStopClicked);
        connect(m_previewBtn, &QPushButton::clicked, this, &MainWindow::onPreviewCurrentFrameClicked);
        connect(m_timelineSlider, &QSlider::sliderReleased, this, [this]() {
            onTimelineScrubbed(m_timelineSlider->value());
        });

        connect(m_engine, &ProcessEngine::logLine, this, &MainWindow::appendEngineLog);
        connect(m_engine, &ProcessEngine::fileProgressUpdated, this, &MainWindow::onFileProgress);
        connect(m_engine, &ProcessEngine::batchProgressUpdated, this, &MainWindow::onBatchProgress);
        connect(m_engine, &ProcessEngine::processingFinished, this, &MainWindow::onProcessingFinished);
    }

    void appendLog(const QString &line)
    {
        if (!m_logConsole) return;
        m_logConsole->appendPlainText(line);
        auto *sb = m_logConsole->verticalScrollBar();
        sb->setValue(sb->maximum());
    }

    void setPreviewPixmap(QLabel *target, const QString &path)
    {
        if (!target) return;
        QPixmap px(path);
        if (px.isNull()) {
            target->setText(QStringLiteral("Preview unavailable"));
            target->setPixmap(QPixmap());
            return;
        }
        target->setPixmap(px.scaled(target->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        target->setText(QString());
    }

    bool extractFrameAt(const QString &videoPath, double seconds, const QString &outputPng)
    {
        if (videoPath.trimmed().isEmpty() || !QFileInfo::exists(videoPath)) return false;

        if (isImagePath(videoPath)) {
            if (videoPath == outputPng) return QFileInfo::exists(videoPath);
            QFile::remove(outputPng);
            return QFile::copy(videoPath, outputPng);
        }

        QProcess ffmpeg;
        ffmpeg.start(QStringLiteral("ffmpeg"), ffmpegExtractArgs(videoPath, seconds, outputPng));
        if (!ffmpeg.waitForFinished(12000)) {
            appendLog(QStringLiteral("[WARN] ffmpeg timeout while extracting frame."));
            return false;
        }
        return ffmpeg.exitCode() == 0 && QFileInfo::exists(outputPng);
    }

    double probeDuration(const QString &videoPath)
    {
        if (isImagePath(videoPath)) return 0.0;

        QProcess probe;
        probe.start(QStringLiteral("ffprobe"), {
            QStringLiteral("-v"), QStringLiteral("error"),
            QStringLiteral("-show_entries"), QStringLiteral("format=duration"),
            QStringLiteral("-of"), QStringLiteral("default=noprint_wrappers=1:nokey=1"),
            videoPath
        });
        if (!probe.waitForFinished(5000)) return 0.0;

        bool ok = false;
        const double sec = QString::fromUtf8(probe.readAllStandardOutput()).trimmed().toDouble(&ok);
        return ok ? qMax(0.0, sec) : 0.0;
    }

    void loadFirstFrame(const QString &path)
    {
        if (path.trimmed().isEmpty() || !QFileInfo::exists(path)) {
            appendLog(QStringLiteral("[WARN] loadFirstFrame: invalid path."));
            return;
        }

        m_durationSec = probeDuration(path);

        const QString frame = QDir::temp().absoluteFilePath(
            QStringLiteral("seedvr2_first_frame_%1.png").arg(QDateTime::currentMSecsSinceEpoch()));

        if (!extractFrameAt(path, 0.0, frame)) {
            appendLog(QStringLiteral("[ERROR] Could not extract first frame."));
            return;
        }

        setPreviewPixmap(m_inputPreview, frame);
        setPreviewPixmap(m_outputPreview, frame);
        appendLog(QStringLiteral("[INFO] First frame loaded."));
    }

    void clearThumbnails()
    {
        while (QLayoutItem *it = m_thumbLayout->takeAt(0)) {
            if (it->widget()) it->widget()->deleteLater();
            delete it;
        }
    }

    void buildTimelineThumbnails(const QString &path)
    {
        clearThumbnails();
        if (path.isEmpty() || !QFileInfo::exists(path)) return;

        if (isImagePath(path)) {
            auto *lbl = new QLabel(m_thumbHost);
            lbl->setFixedSize(110, 72);
            lbl->setObjectName(QStringLiteral("previewLabel"));
            setPreviewPixmap(lbl, path);
            m_thumbLayout->addWidget(lbl);
            m_thumbLayout->addStretch(1);
            return;
        }

        const int thumbCount = 12;
        const double dur = qMax(0.001, m_durationSec);

        for (int i = 0; i < thumbCount; ++i) {
            const double t = (dur * i) / qMax(1, thumbCount - 1);
            const QString out = QDir::temp().absoluteFilePath(
                QStringLiteral("seedvr2_tl_%1_%2.png").arg(QDateTime::currentMSecsSinceEpoch()).arg(i));
            if (!extractFrameAt(path, t, out)) continue;

            auto *lbl = new QLabel(m_thumbHost);
            lbl->setFixedSize(110, 72);
            lbl->setObjectName(QStringLiteral("previewLabel"));
            setPreviewPixmap(lbl, out);
            lbl->setToolTip(QStringLiteral("%1 s").arg(QString::number(t, 'f', 2)));
            m_thumbLayout->addWidget(lbl);
        }
        m_thumbLayout->addStretch(1);
    }

    void loadConfigJsonOnStartup()
    {
        const QString configPath = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("config.json"));
        QFile f(configPath);
        if (!f.open(QIODevice::ReadOnly)) {
            appendLog(QStringLiteral("[ERROR] config.json missing: %1 (fallback to manual UI paths)").arg(configPath));
            return;
        }

        QJsonParseError err;
        const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
        if (err.error != QJsonParseError::NoError || !doc.isObject()) {
            appendLog(QStringLiteral("[ERROR] config.json parse failed: %1").arg(err.errorString()));
            return;
        }

        const QJsonObject obj = doc.object();
        const QString seedFolder = obj.value(QStringLiteral("SeedVR2 Folder")).toString().trimmed();
        const QString pythonExe = obj.value(QStringLiteral("Python Executable")).toString().trimmed();
        const QString inputPath = obj.value(QStringLiteral("Input Path")).toString().trimmed();
        const QString outputPath = obj.value(QStringLiteral("Output Path")).toString().trimmed();

        if (!seedFolder.isEmpty()) {
            m_seedFolderEdit->setText(QFileInfo(seedFolder).absoluteFilePath());
            appendLog(QStringLiteral("[INFO] SeedVR2 Folder loaded from config.json: %1").arg(m_seedFolderEdit->text()));
        } else {
            appendLog(QStringLiteral("[ERROR] 'SeedVR2 Folder' key missing in config.json."));
        }

        if (!pythonExe.isEmpty()) m_pythonExeEdit->setText(pythonExe);
        if (!inputPath.isEmpty()) m_inputPathEdit->setText(inputPath);
        if (!outputPath.isEmpty()) m_outputDirEdit->setText(outputPath);

        if (!m_inputPathEdit->text().trimmed().isEmpty() && QFileInfo::exists(m_inputPathEdit->text().trimmed())) {
            m_currentInputPath = m_inputPathEdit->text().trimmed();
            loadFirstFrame(m_currentInputPath);
            buildTimelineThumbnails(m_currentInputPath);
        }
    }

    QString resolveCliScriptAbsolutePath()
    {
        const QString explicitSeed = m_seedFolderEdit->text().trimmed();
        if (!explicitSeed.isEmpty()) {
            const QString candidate = QDir(explicitSeed).absoluteFilePath(QStringLiteral("inference_cli.py"));
            if (QFileInfo::exists(candidate)) return QFileInfo(candidate).absoluteFilePath();
            appendLog(QStringLiteral("[ERROR] Invalid SeedVR2 Folder path (inference_cli.py missing): %1").arg(explicitSeed));
        }

        // Fallback: re-read config.json only as backup.
        const QString cfg = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("config.json"));
        QFile f(cfg);
        if (f.open(QIODevice::ReadOnly)) {
            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
            if (doc.isObject()) {
                const QString seed = doc.object().value(QStringLiteral("SeedVR2 Folder")).toString().trimmed();
                const QString candidate = QDir(seed).absoluteFilePath(QStringLiteral("inference_cli.py"));
                if (!seed.isEmpty() && QFileInfo::exists(candidate)) {
                    return QFileInfo(candidate).absoluteFilePath();
                }
            }
        }

        appendLog(QStringLiteral("[ERROR] Could not resolve inference_cli.py from config.json or sidebar path."));
        return {};
    }

    QJsonArray buildFfmpegArgs() const
    {
        QJsonArray a;

        const QString codec = m_codecCombo->currentText();
        if (codec == QStringLiteral("H265")) {
            a.append(QStringLiteral("-c:v")); a.append(QStringLiteral("libx265"));
        } else if (codec == QStringLiteral("H264")) {
            a.append(QStringLiteral("-c:v")); a.append(QStringLiteral("libx264"));
        } else if (codec == QStringLiteral("AV1")) {
            a.append(QStringLiteral("-c:v")); a.append(QStringLiteral("libsvtav1"));
        } else if (codec == QStringLiteral("VP9")) {
            a.append(QStringLiteral("-c:v")); a.append(QStringLiteral("libvpx-vp9"));
        }

        if (m_10bitCheck->isChecked()) {
            a.append(QStringLiteral("-pix_fmt"));
            a.append(QStringLiteral("yuv420p10le"));
        }

        a.append(QStringLiteral("-preset"));
        a.append(m_presetCombo->currentText());

        if (m_bitrateModeCombo->currentText() == QStringLiteral("Constant")) {
            const QString br = QString::number(m_constantBitrateSpin->value()) + QStringLiteral("M");
            a.append(QStringLiteral("-b:v")); a.append(br);
            a.append(QStringLiteral("-maxrate")); a.append(br);
            a.append(QStringLiteral("-bufsize")); a.append(QString::number(m_constantBitrateSpin->value() * 2) + QStringLiteral("M"));
        } else {
            a.append(QStringLiteral("-crf"));
            a.append(QString::number(m_crfSpin->value()));
        }

        return a;
    }

private:
    ProcessEngine *m_engine = nullptr;

    // Preview/timeline
    QLabel *m_inputPreview = nullptr;
    QLabel *m_outputPreview = nullptr;
    QSlider *m_timelineSlider = nullptr;
    QWidget *m_thumbHost = nullptr;
    QHBoxLayout *m_thumbLayout = nullptr;

    // Sidebar widgets
    QToolBox *m_toolbox = nullptr;
    QLineEdit *m_inputPathEdit = nullptr;
    QLineEdit *m_outputDirEdit = nullptr;
    QLineEdit *m_seedFolderEdit = nullptr;
    QLineEdit *m_pythonExeEdit = nullptr;
    QLineEdit *m_lutEdit = nullptr;

    QSpinBox *m_resolutionSpin = nullptr;
    QSpinBox *m_batchSizeSpin = nullptr;
    QSlider *m_inputNoiseSlider = nullptr;
    QSlider *m_latentNoiseSlider = nullptr;

    QComboBox *m_containerCombo = nullptr;
    QComboBox *m_codecCombo = nullptr;
    QComboBox *m_backendCombo = nullptr;
    QComboBox *m_bitrateModeCombo = nullptr;
    QSpinBox *m_constantBitrateSpin = nullptr;
    QSpinBox *m_crfSpin = nullptr;
    QComboBox *m_presetCombo = nullptr;
    QCheckBox *m_10bitCheck = nullptr;

    QPushButton *m_previewBtn = nullptr;
    QPushButton *m_runBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;

    QProgressBar *m_fileProgress = nullptr;
    QProgressBar *m_batchProgress = nullptr;

    // Log dock
    QDockWidget *m_logDock = nullptr;
    QPlainTextEdit *m_logConsole = nullptr;

    // Runtime state
    QString m_currentInputPath;
    QString m_targetOutputPath;
    double m_durationSec = 0.0;
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("SeedVR2 Runner"));
    app.setOrganizationName(QStringLiteral("SeedVR2"));
    app.setApplicationVersion(QStringLiteral("2.5"));

    MainWindow w;
    w.show();

    return app.exec();
}
