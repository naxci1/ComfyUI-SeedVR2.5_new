#include "ProcessEngine.h"

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <QtGlobal>

namespace {

constexpr const char *kStyleSheet = R"(
QWidget { background-color:#161922; color:#e5ebff; font-family:"Segoe UI","Inter",sans-serif; font-size:13px; }
QMainWindow { background-color:#0f1219; }
QFrame#previewPanel, QFrame#settingsPanel, QFrame#logPanel, QFrame#progressPanel { background-color:#1e2331; border:1px solid #2f3852; border-radius:12px; }
QLabel#panelTitle { font-size:16px; font-weight:700; color:#f2f6ff; }
QLineEdit, QComboBox, QSpinBox { background-color:#111726; border:1px solid #384666; border-radius:8px; padding:6px 8px; }
QComboBox::drop-down { border:none; }
QPushButton { background-color:#2e3b5e; border:1px solid #445784; border-radius:8px; padding:6px 12px; color:#edf3ff; font-weight:600; }
QPushButton:hover { background-color:#374a76; }
QPushButton#startButton { background-color:#146dff; border:none; color:white; font-size:14px; }
QPushButton#stopButton { background-color:#8b2c40; border:none; color:white; font-size:14px; }
QTabWidget::pane { border:1px solid #2f3852; border-radius:10px; top:-1px; }
QTabBar::tab { background:#111726; border:1px solid #2f3852; padding:8px 14px; margin-right:4px; border-top-left-radius:8px; border-top-right-radius:8px; }
QTabBar::tab:selected { background:#243151; color:#ffffff; }
QProgressBar { background-color:#0f1523; border:1px solid #32446c; border-radius:8px; color:#d9ecff; text-align:center; min-height:22px; }
QProgressBar::chunk { border-radius:8px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #03ffd5, stop:0.5 #2f8cff, stop:1 #7a5cff); }
QPlainTextEdit { background-color:#0b1019; border:1px solid #2d3958; border-radius:10px; color:#dbe8ff; font-family:"Consolas","JetBrains Mono",monospace; font-size:12px; padding:6px; }
QLabel#previewBox { background:#0d1119; border:1px solid #2f3852; border-radius:10px; color:#9db1da; }
QSplitter::handle { background:#2b3348; }
)";

QWidget *buildPathRow(QWidget *parent, QLineEdit *lineEdit, QPushButton *browseButton)
{
    auto *row = new QWidget(parent);
    auto *layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(lineEdit, 1);
    layout->addWidget(browseButton);
    return row;
}

QString findDefaultCliScriptPath()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).absoluteFilePath(QStringLiteral("inference_cli.py")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../inference_cli.py")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../../inference_cli.py")),
        QDir::current().absoluteFilePath(QStringLiteral("inference_cli.py"))
    };

    for (const QString &candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return candidates.first();
}

void setPreviewPixmap(QLabel *target, const QString &path)
{
    if (!target) {
        return;
    }
    if (path.trimmed().isEmpty() || !QFileInfo::exists(path)) {
        target->setText(QStringLiteral("No preview available"));
        target->setPixmap(QPixmap());
        return;
    }
    const QPixmap px(path);
    if (px.isNull()) {
        target->setText(QStringLiteral("Unable to decode image/video frame"));
        target->setPixmap(QPixmap());
        return;
    }
    target->setPixmap(px.scaled(target->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    target->setText(QString());
}

} // namespace

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr)
        : QMainWindow(parent)
        , m_engine(new ProcessEngine(this))
    {
        setWindowTitle(QStringLiteral("SeedVR2 Legacy Logic Runner (Qt6)"));
        setMinimumSize(1500, 920);

        if (qApp) {
            qApp->setStyleSheet(QString::fromLatin1(kStyleSheet));
        }

        buildUi();
        loadSettings();

        connect(m_engine, &ProcessEngine::logLine, this, &MainWindow::appendLogLine);
        connect(m_engine, &ProcessEngine::fileProgressUpdated, this, &MainWindow::onFileProgress);
        connect(m_engine, &ProcessEngine::batchProgressUpdated, this, &MainWindow::onBatchProgress);
        connect(m_engine, &ProcessEngine::processingFinished, this, &MainWindow::onProcessingFinished);

        setRunning(false);
    }

    ~MainWindow() override = default;

protected:
    void closeEvent(QCloseEvent *event) override
    {
        saveSettings();
        if (m_engine->isRunning()) {
            m_engine->stopProcess();
        }
        QMainWindow::closeEvent(event);
    }

private slots:
    void browseInputPath()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Input Media"),
            m_inputPath->text(),
            QStringLiteral("Media (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));

        if (path.isEmpty()) {
            const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Or Select Input Folder"), m_inputPath->text());
            if (!dir.isEmpty()) {
                m_inputPath->setText(dir);
            }
            return;
        }

        m_inputPath->setText(path);
        m_previewInputPath->setText(path);
        refreshPreviewPanels();
    }

    void browseOutputPath()
    {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), m_outputPath->text());
        if (!dir.isEmpty()) {
            m_outputPath->setText(dir);
        }
    }

    void browsePythonExecutable()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Python Executable"),
            m_pythonPath->text(),
            QStringLiteral("Executables (*.exe python python3);;All files (*)"));
        if (!path.isEmpty()) {
            m_pythonPath->setText(path);
        }
    }

    void browseModelDir()
    {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Model Directory"), m_modelDirPath->text());
        if (!dir.isEmpty()) {
            m_modelDirPath->setText(dir);
        }
    }

    void browsePreviewInput()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Input Preview Image"),
            m_previewInputPath->text(),
            QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
        if (!path.isEmpty()) {
            m_previewInputPath->setText(path);
            refreshPreviewPanels();
        }
    }

    void browsePreviewOutput()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Output Preview Image"),
            m_previewOutputPath->text(),
            QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
        if (!path.isEmpty()) {
            m_previewOutputPath->setText(path);
            refreshPreviewPanels();
        }
    }

    void refreshPreviewPanels()
    {
        setPreviewPixmap(m_splitInputPreview, m_previewInputPath->text().trimmed());
        setPreviewPixmap(m_splitOutputPreview, m_previewOutputPath->text().trimmed());
        setPreviewPixmap(m_inputOnlyPreview, m_previewInputPath->text().trimmed());
        setPreviewPixmap(m_outputOnlyPreview, m_previewOutputPath->text().trimmed());
    }

    void updateResolutionMode(const QString &mode)
    {
        const bool pixel = (mode == QStringLiteral("Pixel"));
        m_resolutionSpin->setVisible(pixel);
        m_resolutionTimesCombo->setVisible(!pixel);
    }

    void startRender()
    {
        const QString input = m_inputPath->text().trimmed();
        const QString output = m_outputPath->text().trimmed();

        if (input.isEmpty()) {
            QMessageBox::warning(this,
                                 QStringLiteral("Missing Fields"),
                                 QStringLiteral("Please provide Input path before starting."));
            return;
        }

        const QString scriptPath = findDefaultCliScriptPath();
        if (!QFileInfo::exists(scriptPath)) {
            QMessageBox::critical(this,
                                  QStringLiteral("CLI Not Found"),
                                  QStringLiteral("Could not locate inference_cli.py near the executable or repository root."));
            return;
        }

        const QString pythonExe = m_pythonPath->text().trimmed().isEmpty()
            ? m_pythonExecutable
            : m_pythonPath->text().trimmed();

        saveSettings();

        m_logConsole->clear();
        m_currentFileLabel->setText(QStringLiteral("Current File: — (0/0)"));
        m_batchLabel->setText(QStringLiteral("Batch Progress: 0/0"));
        m_fileProgressBar->setValue(0);
        m_batchProgressBar->setValue(0);

        QStringList args;
        args << input;

        if (!output.isEmpty()) {
            args << QStringLiteral("--output") << output;
        }

        const QString modelDir = m_modelDirPath->text().trimmed();
        if (!modelDir.isEmpty()) {
            args << QStringLiteral("--model_dir") << modelDir;
        }

        args << QStringLiteral("--output_format") << m_outputFormatCombo->currentText().toLower();
        args << QStringLiteral("--video_backend") << m_videoBackendCombo->currentText().toLower();

        const QString ffmpegArgs = m_ffmpegArgsEdit->text().trimmed();
        if (m_videoBackendCombo->currentText().toLower() == QStringLiteral("ffmpeg") && !ffmpegArgs.isEmpty()) {
            args << QStringLiteral("--ffmpeg_video_args") << ffmpegArgs;
        }

        if (m_use10bitCheck->isChecked()) {
            args << QStringLiteral("--10bit");
        }

        args << QStringLiteral("--dit_model") << m_ditModelCombo->currentText();

        const int preDownscale = m_preDownscaleCombo->currentText().left(1).toInt();
        if (preDownscale > 1) {
            args << QStringLiteral("--pre_downscale") << QString::number(preDownscale);
        }

        if (m_resolutionModeCombo->currentText() == QStringLiteral("X Times")) {
            args << QStringLiteral("--resolution_mode") << QStringLiteral("xtimes");
            const int times = m_resolutionTimesCombo->currentText().left(1).toInt();
            args << QStringLiteral("--resolution_scale") << QString::number(times);
        } else {
            args << QStringLiteral("--resolution") << QString::number(m_resolutionSpin->value());
        }

        if (m_maxResolutionSpin->value() > 0) {
            args << QStringLiteral("--max_resolution") << QString::number(m_maxResolutionSpin->value());
        }

        args << QStringLiteral("--batch_size") << QString::number(m_batchSizeSpin->value());

        if (m_uniformBatchCheck->isChecked()) {
            args << QStringLiteral("--uniform_batch_size");
        }

        args << QStringLiteral("--seed") << QString::number(m_seedSpin->value());

        if (m_skipFirstFramesSpin->value() > 0) {
            args << QStringLiteral("--skip_first_frames") << QString::number(m_skipFirstFramesSpin->value());
        }
        if (m_loadCapSpin->value() > 0) {
            args << QStringLiteral("--load_cap") << QString::number(m_loadCapSpin->value());
        }
        if (m_chunkSizeSpin->value() > 0) {
            args << QStringLiteral("--chunk_size") << QString::number(m_chunkSizeSpin->value());
        }
        if (m_prependFramesSpin->value() > 0) {
            args << QStringLiteral("--prepend_frames") << QString::number(m_prependFramesSpin->value());
        }
        if (m_temporalOverlapSpin->value() > 0) {
            args << QStringLiteral("--temporal_overlap") << QString::number(m_temporalOverlapSpin->value());
        }

        if (m_colorCorrectionCombo->currentText() != QStringLiteral("lab")) {
            args << QStringLiteral("--color_correction") << m_colorCorrectionCombo->currentText();
        }

        args << QStringLiteral("--cuda_device") << m_cudaDeviceEdit->text().trimmed();

        if (m_ditOffloadCombo->currentText() != QStringLiteral("none")) {
            args << QStringLiteral("--dit_offload_device") << m_ditOffloadCombo->currentText();
        }
        if (m_vaeOffloadCombo->currentText() != QStringLiteral("none")) {
            args << QStringLiteral("--vae_offload_device") << m_vaeOffloadCombo->currentText();
        }
        if (m_tensorOffloadCombo->currentText() != QStringLiteral("none")) {
            args << QStringLiteral("--tensor_offload_device") << m_tensorOffloadCombo->currentText();
        }

        if (m_blocksToSwapSpin->value() > 0) {
            args << QStringLiteral("--blocks_to_swap") << QString::number(m_blocksToSwapSpin->value());
        }
        if (m_swapIoComponentsCheck->isChecked()) {
            args << QStringLiteral("--swap_io_components");
        }

        if (m_vaeEncodeTiledCheck->isChecked()) {
            args << QStringLiteral("--vae_encode_tiled");
            if (m_vaeEncodeTileSizeSpin->value() != 1024) {
                args << QStringLiteral("--vae_encode_tile_size") << QString::number(m_vaeEncodeTileSizeSpin->value());
            }
            if (m_vaeEncodeTileOverlapSpin->value() != 128) {
                args << QStringLiteral("--vae_encode_tile_overlap") << QString::number(m_vaeEncodeTileOverlapSpin->value());
            }
        }

        if (m_vaeDecodeTiledCheck->isChecked()) {
            args << QStringLiteral("--vae_decode_tiled");
            if (m_vaeDecodeTileSizeSpin->value() != 1024) {
                args << QStringLiteral("--vae_decode_tile_size") << QString::number(m_vaeDecodeTileSizeSpin->value());
            }
            if (m_vaeDecodeTileOverlapSpin->value() != 128) {
                args << QStringLiteral("--vae_decode_tile_overlap") << QString::number(m_vaeDecodeTileOverlapSpin->value());
            }
        }

        if (m_tileDebugCombo->currentText() != QStringLiteral("false")) {
            args << QStringLiteral("--tile_debug") << m_tileDebugCombo->currentText();
        }

        if (m_attentionModeCombo->currentText() != QStringLiteral("sdpa")) {
            args << QStringLiteral("--attention_mode") << m_attentionModeCombo->currentText();
        }

        if (m_cacheDitCheck->isChecked()) {
            args << QStringLiteral("--cache_dit");
        }
        if (m_cacheVaeCheck->isChecked()) {
            args << QStringLiteral("--cache_vae");
        }
        if (m_autoSafeguardCheck->isChecked()) {
            args << QStringLiteral("--auto_safeguard");
        }
        if (m_debugCheck->isChecked()) {
            args << QStringLiteral("--debug");
        }

        setRunning(true);
        statusBar()->showMessage(QStringLiteral("Rendering started..."));
        m_engine->startProcess(pythonExe, scriptPath, args);
    }

    void stopRender()
    {
        m_engine->stopProcess();
        statusBar()->showMessage(QStringLiteral("Stopping render..."));
    }

    void appendLogLine(const QString &line)
    {
        m_logConsole->appendPlainText(line);
        QScrollBar *scroll = m_logConsole->verticalScrollBar();
        scroll->setValue(scroll->maximum());
    }

    void onFileProgress(const QString &filename, int current, int total, int doneFiles, int remainingFiles, int)
    {
        const int filePct = (total > 0) ? qBound(0, (current * 100) / total, 100) : 0;
        m_fileProgressBar->setValue(filePct);
        m_currentFileLabel->setText(
            QStringLiteral("Current File: %1 (%2/%3)")
                .arg(QFileInfo(filename).fileName().isEmpty() ? QStringLiteral("—") : QFileInfo(filename).fileName())
                .arg(current)
                .arg(total));

        const int totalFiles = doneFiles + remainingFiles + 1;
        const int batchPct = (totalFiles > 0) ? qBound(0, (doneFiles * 100) / totalFiles, 100) : 0;
        m_batchProgressBar->setValue(batchPct);
        m_batchLabel->setText(QStringLiteral("Batch Progress: %1/%2").arg(doneFiles).arg(totalFiles));

        const QString outputPreview = m_previewOutputPath->text().trimmed();
        if (!outputPreview.isEmpty() && QFileInfo::exists(outputPreview)) {
            refreshPreviewPanels();
        }
    }

    void onBatchProgress(int current, int total)
    {
        if (total > 0) {
            m_batchProgressBar->setValue(qBound(0, (current * 100) / total, 100));
            m_batchLabel->setText(QStringLiteral("Batch Progress: %1/%2").arg(current).arg(total));
        }
    }

    void onProcessingFinished(bool success, const QString &message)
    {
        setRunning(false);
        statusBar()->showMessage(success ? QStringLiteral("Render complete.") : QStringLiteral("Render stopped."), 5000);
        appendLogLine(QStringLiteral("Process result: %1").arg(message));

        if (success) {
            m_fileProgressBar->setValue(100);
            m_batchProgressBar->setValue(100);
            refreshPreviewPanels();
        }
    }

private:
    void setRunning(bool running)
    {
        m_startButton->setEnabled(!running);
        m_stopButton->setEnabled(running);
    }

    void buildUi()
    {
        auto *central = new QWidget(this);
        setCentralWidget(central);

        auto *rootLayout = new QVBoxLayout(central);
        rootLayout->setContentsMargins(12, 12, 12, 12);
        rootLayout->setSpacing(10);

        auto *mainSplitter = new QSplitter(Qt::Horizontal, central);
        rootLayout->addWidget(mainSplitter);

        auto *leftPanel = new QWidget(mainSplitter);
        auto *leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        leftLayout->setSpacing(10);

        auto *previewFrame = new QFrame(leftPanel);
        previewFrame->setObjectName(QStringLiteral("previewPanel"));
        auto *previewLayout = new QVBoxLayout(previewFrame);
        previewLayout->setContentsMargins(12, 12, 12, 12);
        previewLayout->setSpacing(8);

        auto *previewTitle = new QLabel(QStringLiteral("Preview Workspace"), previewFrame);
        previewTitle->setObjectName(QStringLiteral("panelTitle"));
        previewLayout->addWidget(previewTitle);

        auto *previewSelector = new QGroupBox(QStringLiteral("Preview Sources"), previewFrame);
        auto *previewSelectorLayout = new QFormLayout(previewSelector);

        m_previewInputPath = new QLineEdit(previewSelector);
        m_previewInputBrowseButton = new QPushButton(QStringLiteral("Browse"), previewSelector);
        previewSelectorLayout->addRow(QStringLiteral("Input Frame:"), buildPathRow(previewSelector, m_previewInputPath, m_previewInputBrowseButton));

        m_previewOutputPath = new QLineEdit(previewSelector);
        m_previewOutputBrowseButton = new QPushButton(QStringLiteral("Browse"), previewSelector);
        previewSelectorLayout->addRow(QStringLiteral("Upscaled Frame:"), buildPathRow(previewSelector, m_previewOutputPath, m_previewOutputBrowseButton));

        auto *previewRefreshButton = new QPushButton(QStringLiteral("Refresh Preview"), previewSelector);
        previewSelectorLayout->addRow(QString(), previewRefreshButton);
        previewLayout->addWidget(previewSelector);

        m_previewTabs = new QTabWidget(previewFrame);

        auto *splitTab = new QWidget(m_previewTabs);
        auto *splitTabLayout = new QVBoxLayout(splitTab);
        auto *splitter = new QSplitter(Qt::Horizontal, splitTab);

        m_splitInputPreview = new QLabel(QStringLiteral("Original"), splitter);
        m_splitInputPreview->setAlignment(Qt::AlignCenter);
        m_splitInputPreview->setObjectName(QStringLiteral("previewBox"));
        m_splitInputPreview->setMinimumSize(400, 320);

        m_splitOutputPreview = new QLabel(QStringLiteral("Upscaled"), splitter);
        m_splitOutputPreview->setAlignment(Qt::AlignCenter);
        m_splitOutputPreview->setObjectName(QStringLiteral("previewBox"));
        m_splitOutputPreview->setMinimumSize(400, 320);

        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 1);
        splitTabLayout->addWidget(splitter);
        m_previewTabs->addTab(splitTab, QStringLiteral("Split View"));

        auto *inputTab = new QWidget(m_previewTabs);
        auto *inputTabLayout = new QVBoxLayout(inputTab);
        m_inputOnlyPreview = new QLabel(QStringLiteral("Input Preview"), inputTab);
        m_inputOnlyPreview->setAlignment(Qt::AlignCenter);
        m_inputOnlyPreview->setObjectName(QStringLiteral("previewBox"));
        m_inputOnlyPreview->setMinimumSize(800, 360);
        inputTabLayout->addWidget(m_inputOnlyPreview);
        m_previewTabs->addTab(inputTab, QStringLiteral("Input"));

        auto *outputTab = new QWidget(m_previewTabs);
        auto *outputTabLayout = new QVBoxLayout(outputTab);
        m_outputOnlyPreview = new QLabel(QStringLiteral("Output Preview"), outputTab);
        m_outputOnlyPreview->setAlignment(Qt::AlignCenter);
        m_outputOnlyPreview->setObjectName(QStringLiteral("previewBox"));
        m_outputOnlyPreview->setMinimumSize(800, 360);
        outputTabLayout->addWidget(m_outputOnlyPreview);
        m_previewTabs->addTab(outputTab, QStringLiteral("Output"));

        previewLayout->addWidget(m_previewTabs, 1);
        leftLayout->addWidget(previewFrame, 4);

        auto *progressFrame = new QFrame(leftPanel);
        progressFrame->setObjectName(QStringLiteral("progressPanel"));
        auto *progressLayout = new QVBoxLayout(progressFrame);
        progressLayout->setContentsMargins(12, 12, 12, 12);
        progressLayout->setSpacing(6);

        auto *progressTitle = new QLabel(QStringLiteral("Progress"), progressFrame);
        progressTitle->setObjectName(QStringLiteral("panelTitle"));
        progressLayout->addWidget(progressTitle);

        m_currentFileLabel = new QLabel(QStringLiteral("Current File: — (0/0)"), progressFrame);
        m_fileProgressBar = new QProgressBar(progressFrame);
        m_fileProgressBar->setRange(0, 100);
        m_fileProgressBar->setValue(0);

        m_batchLabel = new QLabel(QStringLiteral("Batch Progress: 0/0"), progressFrame);
        m_batchProgressBar = new QProgressBar(progressFrame);
        m_batchProgressBar->setRange(0, 100);
        m_batchProgressBar->setValue(0);

        progressLayout->addWidget(m_currentFileLabel);
        progressLayout->addWidget(m_fileProgressBar);
        progressLayout->addWidget(m_batchLabel);
        progressLayout->addWidget(m_batchProgressBar);
        leftLayout->addWidget(progressFrame, 0);

        auto *logFrame = new QFrame(leftPanel);
        logFrame->setObjectName(QStringLiteral("logPanel"));
        auto *logLayout = new QVBoxLayout(logFrame);
        logLayout->setContentsMargins(12, 12, 12, 12);
        logLayout->setSpacing(8);

        auto *logTitle = new QLabel(QStringLiteral("Real-time Inference Log"), logFrame);
        logTitle->setObjectName(QStringLiteral("panelTitle"));
        m_logConsole = new QPlainTextEdit(logFrame);
        m_logConsole->setReadOnly(true);
        m_logConsole->setMaximumBlockCount(10000);
        m_logConsole->setMinimumHeight(210);

        logLayout->addWidget(logTitle);
        logLayout->addWidget(m_logConsole, 1);
        leftLayout->addWidget(logFrame, 2);

        auto *rightFrame = new QFrame(mainSplitter);
        rightFrame->setObjectName(QStringLiteral("settingsPanel"));
        auto *rightLayout = new QVBoxLayout(rightFrame);
        rightLayout->setContentsMargins(12, 12, 12, 12);
        rightLayout->setSpacing(10);

        auto *settingsTitle = new QLabel(QStringLiteral("Legacy Parameter Mapping"), rightFrame);
        settingsTitle->setObjectName(QStringLiteral("panelTitle"));
        rightLayout->addWidget(settingsTitle);

        auto *scrollArea = new QScrollArea(rightFrame);
        scrollArea->setWidgetResizable(true);
        scrollArea->setFrameShape(QFrame::NoFrame);

        auto *scrollHost = new QWidget(scrollArea);
        auto *scrollLayout = new QVBoxLayout(scrollHost);
        scrollLayout->setContentsMargins(0, 0, 0, 0);
        scrollLayout->setSpacing(8);

        auto *pathsGroup = new QGroupBox(QStringLiteral("Paths"), scrollHost);
        auto *pathsForm = new QFormLayout(pathsGroup);
        m_pythonPath = new QLineEdit(pathsGroup);
        m_pythonBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        pathsForm->addRow(QStringLiteral("Python Executable:"), buildPathRow(pathsGroup, m_pythonPath, m_pythonBrowseButton));

        m_inputPath = new QLineEdit(pathsGroup);
        m_inputBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        pathsForm->addRow(QStringLiteral("Input (file/folder):"), buildPathRow(pathsGroup, m_inputPath, m_inputBrowseButton));

        m_outputPath = new QLineEdit(pathsGroup);
        m_outputBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        pathsForm->addRow(QStringLiteral("Output:"), buildPathRow(pathsGroup, m_outputPath, m_outputBrowseButton));

        m_modelDirPath = new QLineEdit(pathsGroup);
        m_modelDirBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        pathsForm->addRow(QStringLiteral("Model Directory:"), buildPathRow(pathsGroup, m_modelDirPath, m_modelDirBrowseButton));
        scrollLayout->addWidget(pathsGroup);

        auto *modelGroup = new QGroupBox(QStringLiteral("AI Model"), scrollHost);
        auto *modelForm = new QFormLayout(modelGroup);
        m_ditModelCombo = new QComboBox(modelGroup);
        m_ditModelCombo->addItems({
            QStringLiteral("seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
            QStringLiteral("seedvr2_ema_3b-Q4_K_M.gguf"),
            QStringLiteral("seedvr2_ema_3b-Q8_0.gguf"),
            QStringLiteral("seedvr2_ema_3b_fp16.safetensors"),
            QStringLiteral("seedvr2_ema_7b-Q4_K_M.gguf"),
            QStringLiteral("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"),
            QStringLiteral("seedvr2_ema_7b_fp16.safetensors"),
            QStringLiteral("seedvr2_ema_7b_sharp-Q4_K_M.gguf"),
            QStringLiteral("seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors"),
            QStringLiteral("seedvr2_ema_7b_sharp_fp16.safetensors")
        });
        m_ditModelCombo->setCurrentText(QStringLiteral("seedvr2_ema_3b-Q8_0.gguf"));
        modelForm->addRow(QStringLiteral("DiT Model:"), m_ditModelCombo);
        scrollLayout->addWidget(modelGroup);

        auto *procGroup = new QGroupBox(QStringLiteral("Processing Settings"), scrollHost);
        auto *procForm = new QFormLayout(procGroup);
        m_preDownscaleCombo = new QComboBox(procGroup);
        m_preDownscaleCombo->addItems({ QStringLiteral("1:1"), QStringLiteral("2:1"), QStringLiteral("3:1") });
        procForm->addRow(QStringLiteral("Pre-Downscale:"), m_preDownscaleCombo);

        auto *resRow = new QWidget(procGroup);
        auto *resRowLayout = new QHBoxLayout(resRow);
        resRowLayout->setContentsMargins(0, 0, 0, 0);
        m_resolutionModeCombo = new QComboBox(procGroup);
        m_resolutionModeCombo->addItems({ QStringLiteral("Pixel"), QStringLiteral("X Times") });
        m_resolutionSpin = new QSpinBox(procGroup);
        m_resolutionSpin->setRange(128, 7680);
        m_resolutionSpin->setValue(720);
        m_resolutionTimesCombo = new QComboBox(procGroup);
        m_resolutionTimesCombo->addItems({ QStringLiteral("1x"), QStringLiteral("2x"), QStringLiteral("3x"), QStringLiteral("4x"), QStringLiteral("5x") });
        m_resolutionTimesCombo->setCurrentText(QStringLiteral("2x"));
        resRowLayout->addWidget(m_resolutionModeCombo);
        resRowLayout->addWidget(m_resolutionSpin);
        resRowLayout->addWidget(m_resolutionTimesCombo);
        procForm->addRow(QStringLiteral("Resolution:"), resRow);

        m_maxResolutionSpin = new QSpinBox(procGroup);
        m_maxResolutionSpin->setRange(0, 7680);
        m_maxResolutionSpin->setValue(0);
        procForm->addRow(QStringLiteral("Max Resolution:"), m_maxResolutionSpin);

        m_batchSizeSpin = new QSpinBox(procGroup);
        m_batchSizeSpin->setRange(1, 10001);
        m_batchSizeSpin->setValue(81);
        procForm->addRow(QStringLiteral("Batch Size:"), m_batchSizeSpin);

        m_uniformBatchCheck = new QCheckBox(procGroup);
        procForm->addRow(QStringLiteral("Uniform Batch Size:"), m_uniformBatchCheck);

        m_temporalOverlapSpin = new QSpinBox(procGroup);
        m_temporalOverlapSpin->setRange(0, 100);
        procForm->addRow(QStringLiteral("Temporal Overlap:"), m_temporalOverlapSpin);

        m_prependFramesSpin = new QSpinBox(procGroup);
        m_prependFramesSpin->setRange(0, 100);
        procForm->addRow(QStringLiteral("Prepend Frames:"), m_prependFramesSpin);
        scrollLayout->addWidget(procGroup);

        auto *previewGroup = new QGroupBox(QStringLiteral("Preview & Processing"), scrollHost);
        auto *previewForm = new QFormLayout(previewGroup);
        m_seedSpin = new QSpinBox(previewGroup);
        m_seedSpin->setRange(0, 2147483647);
        m_seedSpin->setValue(313);
        previewForm->addRow(QStringLiteral("Seed:"), m_seedSpin);

        m_skipFirstFramesSpin = new QSpinBox(previewGroup);
        m_skipFirstFramesSpin->setRange(0, 99999);
        previewForm->addRow(QStringLiteral("Skip First Frames:"), m_skipFirstFramesSpin);

        m_loadCapSpin = new QSpinBox(previewGroup);
        m_loadCapSpin->setRange(0, 99999);
        previewForm->addRow(QStringLiteral("Load Cap:"), m_loadCapSpin);

        m_chunkSizeSpin = new QSpinBox(previewGroup);
        m_chunkSizeSpin->setRange(0, 99999);
        previewForm->addRow(QStringLiteral("Chunk Size:"), m_chunkSizeSpin);
        scrollLayout->addWidget(previewGroup);

        auto *deviceGroup = new QGroupBox(QStringLiteral("Device Management"), scrollHost);
        auto *deviceForm = new QFormLayout(deviceGroup);

        m_cudaDeviceEdit = new QLineEdit(deviceGroup);
        m_cudaDeviceEdit->setText(QStringLiteral("0"));
        deviceForm->addRow(QStringLiteral("CUDA Device(s):"), m_cudaDeviceEdit);

        m_ditOffloadCombo = new QComboBox(deviceGroup);
        m_ditOffloadCombo->addItems({ QStringLiteral("none"), QStringLiteral("cpu"), QStringLiteral("0"), QStringLiteral("1") });
        deviceForm->addRow(QStringLiteral("DiT Offload:"), m_ditOffloadCombo);

        m_vaeOffloadCombo = new QComboBox(deviceGroup);
        m_vaeOffloadCombo->addItems({ QStringLiteral("none"), QStringLiteral("cpu"), QStringLiteral("0"), QStringLiteral("1") });
        deviceForm->addRow(QStringLiteral("VAE Offload:"), m_vaeOffloadCombo);

        m_tensorOffloadCombo = new QComboBox(deviceGroup);
        m_tensorOffloadCombo->addItems({ QStringLiteral("cpu"), QStringLiteral("none"), QStringLiteral("0"), QStringLiteral("1") });
        deviceForm->addRow(QStringLiteral("Tensor Offload:"), m_tensorOffloadCombo);
        scrollLayout->addWidget(deviceGroup);

        auto *qualityGroup = new QGroupBox(QStringLiteral("Quality + Performance"), scrollHost);
        auto *qualityForm = new QFormLayout(qualityGroup);

        m_colorCorrectionCombo = new QComboBox(qualityGroup);
        m_colorCorrectionCombo->addItems({ QStringLiteral("lab"), QStringLiteral("wavelet"), QStringLiteral("wavelet_adaptive"), QStringLiteral("hsv"), QStringLiteral("adain"), QStringLiteral("none") });
        qualityForm->addRow(QStringLiteral("Color Correction:"), m_colorCorrectionCombo);

        m_attentionModeCombo = new QComboBox(qualityGroup);
        m_attentionModeCombo->addItems({ QStringLiteral("sdpa"), QStringLiteral("flash_attn_2"), QStringLiteral("flash_attn_3"), QStringLiteral("sageattn_2"), QStringLiteral("sageattn_3") });
        qualityForm->addRow(QStringLiteral("Attention Mode:"), m_attentionModeCombo);
        scrollLayout->addWidget(qualityGroup);

        auto *memoryGroup = new QGroupBox(QStringLiteral("Memory Optimization + VAE Tiling"), scrollHost);
        auto *memoryForm = new QFormLayout(memoryGroup);

        m_blocksToSwapSpin = new QSpinBox(memoryGroup);
        m_blocksToSwapSpin->setRange(0, 64);
        memoryForm->addRow(QStringLiteral("Blocks To Swap:"), m_blocksToSwapSpin);

        m_swapIoComponentsCheck = new QCheckBox(memoryGroup);
        memoryForm->addRow(QStringLiteral("Swap IO Components:"), m_swapIoComponentsCheck);

        m_vaeEncodeTiledCheck = new QCheckBox(memoryGroup);
        memoryForm->addRow(QStringLiteral("VAE Encode Tiled:"), m_vaeEncodeTiledCheck);

        m_vaeEncodeTileSizeSpin = new QSpinBox(memoryGroup);
        m_vaeEncodeTileSizeSpin->setRange(128, 4096);
        m_vaeEncodeTileSizeSpin->setValue(1024);
        memoryForm->addRow(QStringLiteral("Encode Tile Size:"), m_vaeEncodeTileSizeSpin);

        m_vaeEncodeTileOverlapSpin = new QSpinBox(memoryGroup);
        m_vaeEncodeTileOverlapSpin->setRange(0, 1023);
        m_vaeEncodeTileOverlapSpin->setValue(128);
        memoryForm->addRow(QStringLiteral("Encode Tile Overlap:"), m_vaeEncodeTileOverlapSpin);

        m_vaeDecodeTiledCheck = new QCheckBox(memoryGroup);
        memoryForm->addRow(QStringLiteral("VAE Decode Tiled:"), m_vaeDecodeTiledCheck);

        m_vaeDecodeTileSizeSpin = new QSpinBox(memoryGroup);
        m_vaeDecodeTileSizeSpin->setRange(128, 4096);
        m_vaeDecodeTileSizeSpin->setValue(1024);
        memoryForm->addRow(QStringLiteral("Decode Tile Size:"), m_vaeDecodeTileSizeSpin);

        m_vaeDecodeTileOverlapSpin = new QSpinBox(memoryGroup);
        m_vaeDecodeTileOverlapSpin->setRange(0, 1023);
        m_vaeDecodeTileOverlapSpin->setValue(128);
        memoryForm->addRow(QStringLiteral("Decode Tile Overlap:"), m_vaeDecodeTileOverlapSpin);

        m_tileDebugCombo = new QComboBox(memoryGroup);
        m_tileDebugCombo->addItems({ QStringLiteral("false"), QStringLiteral("encode"), QStringLiteral("decode") });
        memoryForm->addRow(QStringLiteral("Tile Debug:"), m_tileDebugCombo);

        scrollLayout->addWidget(memoryGroup);

        auto *codecGroup = new QGroupBox(QStringLiteral("Codec / Output"), scrollHost);
        auto *codecForm = new QFormLayout(codecGroup);

        m_outputFormatCombo = new QComboBox(codecGroup);
        m_outputFormatCombo->addItems({ QStringLiteral("mp4"), QStringLiteral("mov"), QStringLiteral("mkv"), QStringLiteral("webm"), QStringLiteral("png"), QStringLiteral("tiff"), QStringLiteral("jpg"), QStringLiteral("dpx"), QStringLiteral("exr") });
        codecForm->addRow(QStringLiteral("Output Format:"), m_outputFormatCombo);

        m_videoBackendCombo = new QComboBox(codecGroup);
        m_videoBackendCombo->addItems({ QStringLiteral("ffmpeg"), QStringLiteral("opencv") });
        codecForm->addRow(QStringLiteral("Video Backend:"), m_videoBackendCombo);

        m_ffmpegArgsEdit = new QLineEdit(codecGroup);
        codecForm->addRow(QStringLiteral("FFmpeg Video Args (JSON):"), m_ffmpegArgsEdit);

        m_use10bitCheck = new QCheckBox(codecGroup);
        codecForm->addRow(QStringLiteral("Use 10-bit:"), m_use10bitCheck);

        scrollLayout->addWidget(codecGroup);

        auto *flagsGroup = new QGroupBox(QStringLiteral("Caching / Safety / Debug"), scrollHost);
        auto *flagsForm = new QFormLayout(flagsGroup);
        m_cacheDitCheck = new QCheckBox(flagsGroup);
        flagsForm->addRow(QStringLiteral("Cache DiT:"), m_cacheDitCheck);
        m_cacheVaeCheck = new QCheckBox(flagsGroup);
        flagsForm->addRow(QStringLiteral("Cache VAE:"), m_cacheVaeCheck);
        m_autoSafeguardCheck = new QCheckBox(flagsGroup);
        flagsForm->addRow(QStringLiteral("Auto Safeguard:"), m_autoSafeguardCheck);
        m_debugCheck = new QCheckBox(flagsGroup);
        flagsForm->addRow(QStringLiteral("Debug:"), m_debugCheck);
        scrollLayout->addWidget(flagsGroup);

        scrollLayout->addStretch(1);
        scrollArea->setWidget(scrollHost);
        rightLayout->addWidget(scrollArea, 1);

        auto *buttonRow = new QHBoxLayout();
        m_startButton = new QPushButton(QStringLiteral("Start Render"), rightFrame);
        m_startButton->setObjectName(QStringLiteral("startButton"));
        m_stopButton = new QPushButton(QStringLiteral("Stop"), rightFrame);
        m_stopButton->setObjectName(QStringLiteral("stopButton"));
        buttonRow->addWidget(m_startButton);
        buttonRow->addWidget(m_stopButton);
        rightLayout->addLayout(buttonRow);

        mainSplitter->addWidget(leftPanel);
        mainSplitter->addWidget(rightFrame);
        mainSplitter->setStretchFactor(0, 7);
        mainSplitter->setStretchFactor(1, 3);

        connect(m_inputBrowseButton, &QPushButton::clicked, this, &MainWindow::browseInputPath);
        connect(m_outputBrowseButton, &QPushButton::clicked, this, &MainWindow::browseOutputPath);
        connect(m_pythonBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePythonExecutable);
        connect(m_modelDirBrowseButton, &QPushButton::clicked, this, &MainWindow::browseModelDir);
        connect(m_previewInputBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePreviewInput);
        connect(m_previewOutputBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePreviewOutput);
        connect(previewRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshPreviewPanels);
        connect(m_resolutionModeCombo, &QComboBox::currentTextChanged, this, &MainWindow::updateResolutionMode);

        connect(m_startButton, &QPushButton::clicked, this, &MainWindow::startRender);
        connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::stopRender);

        updateResolutionMode(m_resolutionModeCombo->currentText());
        refreshPreviewPanels();
        statusBar()->showMessage(QStringLiteral("Ready"));
    }

    void saveSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        settings.setValue(QStringLiteral("inputPath"), m_inputPath->text());
        settings.setValue(QStringLiteral("outputPath"), m_outputPath->text());
        settings.setValue(QStringLiteral("modelDirPath"), m_modelDirPath->text());
        settings.setValue(QStringLiteral("pythonPath"), m_pythonPath->text());
        settings.setValue(QStringLiteral("previewInputPath"), m_previewInputPath->text());
        settings.setValue(QStringLiteral("previewOutputPath"), m_previewOutputPath->text());
        settings.setValue(QStringLiteral("ditModel"), m_ditModelCombo->currentText());
        settings.setValue(QStringLiteral("resolutionMode"), m_resolutionModeCombo->currentText());
        settings.setValue(QStringLiteral("resolution"), m_resolutionSpin->value());
        settings.setValue(QStringLiteral("resolutionTimes"), m_resolutionTimesCombo->currentText());
        settings.setValue(QStringLiteral("batchSize"), m_batchSizeSpin->value());
        settings.setValue(QStringLiteral("seed"), m_seedSpin->value());
        settings.setValue(QStringLiteral("cudaDevice"), m_cudaDeviceEdit->text());
    }

    void loadSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        m_inputPath->setText(settings.value(QStringLiteral("inputPath")).toString());
        m_outputPath->setText(settings.value(QStringLiteral("outputPath")).toString());
        m_modelDirPath->setText(settings.value(QStringLiteral("modelDirPath")).toString());
        m_pythonPath->setText(settings.value(QStringLiteral("pythonPath"), QStringLiteral("python")).toString());
        m_previewInputPath->setText(settings.value(QStringLiteral("previewInputPath")).toString());
        m_previewOutputPath->setText(settings.value(QStringLiteral("previewOutputPath")).toString());
        m_ditModelCombo->setCurrentText(settings.value(QStringLiteral("ditModel"), m_ditModelCombo->currentText()).toString());
        m_resolutionModeCombo->setCurrentText(settings.value(QStringLiteral("resolutionMode"), QStringLiteral("Pixel")).toString());
        m_resolutionSpin->setValue(settings.value(QStringLiteral("resolution"), 720).toInt());
        m_resolutionTimesCombo->setCurrentText(settings.value(QStringLiteral("resolutionTimes"), QStringLiteral("2x")).toString());
        m_batchSizeSpin->setValue(settings.value(QStringLiteral("batchSize"), 81).toInt());
        m_seedSpin->setValue(settings.value(QStringLiteral("seed"), 313).toInt());
        m_cudaDeviceEdit->setText(settings.value(QStringLiteral("cudaDevice"), QStringLiteral("0")).toString());

        updateResolutionMode(m_resolutionModeCombo->currentText());
        refreshPreviewPanels();
    }

    QLineEdit *m_inputPath = nullptr;
    QLineEdit *m_outputPath = nullptr;
    QLineEdit *m_modelDirPath = nullptr;
    QLineEdit *m_pythonPath = nullptr;

    QLineEdit *m_previewInputPath = nullptr;
    QLineEdit *m_previewOutputPath = nullptr;

    QPushButton *m_inputBrowseButton = nullptr;
    QPushButton *m_outputBrowseButton = nullptr;
    QPushButton *m_modelDirBrowseButton = nullptr;
    QPushButton *m_pythonBrowseButton = nullptr;
    QPushButton *m_previewInputBrowseButton = nullptr;
    QPushButton *m_previewOutputBrowseButton = nullptr;

    QLabel *m_splitInputPreview = nullptr;
    QLabel *m_splitOutputPreview = nullptr;
    QLabel *m_inputOnlyPreview = nullptr;
    QLabel *m_outputOnlyPreview = nullptr;
    QTabWidget *m_previewTabs = nullptr;

    QComboBox *m_ditModelCombo = nullptr;
    QComboBox *m_preDownscaleCombo = nullptr;
    QComboBox *m_resolutionModeCombo = nullptr;
    QSpinBox *m_resolutionSpin = nullptr;
    QComboBox *m_resolutionTimesCombo = nullptr;
    QSpinBox *m_maxResolutionSpin = nullptr;
    QSpinBox *m_batchSizeSpin = nullptr;
    QCheckBox *m_uniformBatchCheck = nullptr;
    QSpinBox *m_temporalOverlapSpin = nullptr;
    QSpinBox *m_prependFramesSpin = nullptr;

    QSpinBox *m_seedSpin = nullptr;
    QSpinBox *m_skipFirstFramesSpin = nullptr;
    QSpinBox *m_loadCapSpin = nullptr;
    QSpinBox *m_chunkSizeSpin = nullptr;

    QLineEdit *m_cudaDeviceEdit = nullptr;
    QComboBox *m_ditOffloadCombo = nullptr;
    QComboBox *m_vaeOffloadCombo = nullptr;
    QComboBox *m_tensorOffloadCombo = nullptr;

    QComboBox *m_colorCorrectionCombo = nullptr;
    QComboBox *m_attentionModeCombo = nullptr;

    QSpinBox *m_blocksToSwapSpin = nullptr;
    QCheckBox *m_swapIoComponentsCheck = nullptr;
    QCheckBox *m_vaeEncodeTiledCheck = nullptr;
    QSpinBox *m_vaeEncodeTileSizeSpin = nullptr;
    QSpinBox *m_vaeEncodeTileOverlapSpin = nullptr;
    QCheckBox *m_vaeDecodeTiledCheck = nullptr;
    QSpinBox *m_vaeDecodeTileSizeSpin = nullptr;
    QSpinBox *m_vaeDecodeTileOverlapSpin = nullptr;
    QComboBox *m_tileDebugCombo = nullptr;

    QComboBox *m_outputFormatCombo = nullptr;
    QComboBox *m_videoBackendCombo = nullptr;
    QLineEdit *m_ffmpegArgsEdit = nullptr;
    QCheckBox *m_use10bitCheck = nullptr;

    QCheckBox *m_cacheDitCheck = nullptr;
    QCheckBox *m_cacheVaeCheck = nullptr;
    QCheckBox *m_autoSafeguardCheck = nullptr;
    QCheckBox *m_debugCheck = nullptr;

    QPushButton *m_startButton = nullptr;
    QPushButton *m_stopButton = nullptr;

    QLabel *m_currentFileLabel = nullptr;
    QLabel *m_batchLabel = nullptr;
    QProgressBar *m_fileProgressBar = nullptr;
    QProgressBar *m_batchProgressBar = nullptr;

    QPlainTextEdit *m_logConsole = nullptr;

    ProcessEngine *m_engine = nullptr;
    QString m_pythonExecutable = QStringLiteral("python");
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("SeedVR2 Runner"));
    app.setOrganizationName(QStringLiteral("SeedVR2"));
    app.setApplicationVersion(QStringLiteral("2.5"));

    MainWindow window;
    window.show();

    return app.exec();
}
