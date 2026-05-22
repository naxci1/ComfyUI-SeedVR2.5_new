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
#include <QScrollBar>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <QtGlobal>

namespace {

constexpr const char *kStyleSheet = R"(
QWidget {
    background-color: #161922;
    color: #e5ebff;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QMainWindow {
    background-color: #0f1219;
}
QFrame#previewPanel, QFrame#settingsPanel, QFrame#logPanel, QFrame#progressPanel {
    background-color: #1e2331;
    border: 1px solid #2f3852;
    border-radius: 12px;
}
QLabel#panelTitle {
    font-size: 16px;
    font-weight: 700;
    color: #f2f6ff;
}
QLabel#subtleTitle {
    color: #9fb0d3;
    font-size: 12px;
    font-weight: 600;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #111726;
    border: 1px solid #384666;
    border-radius: 8px;
    padding: 6px 8px;
}
QComboBox::drop-down {
    border: none;
}
QPushButton {
    background-color: #2e3b5e;
    border: 1px solid #445784;
    border-radius: 8px;
    padding: 6px 12px;
    color: #edf3ff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #374a76;
}
QPushButton#startButton {
    background-color: #146dff;
    border: none;
    color: white;
    font-size: 14px;
}
QPushButton#stopButton {
    background-color: #8b2c40;
    border: none;
    color: white;
    font-size: 14px;
}
QTabWidget::pane {
    border: 1px solid #2f3852;
    border-radius: 10px;
    top: -1px;
}
QTabBar::tab {
    background: #111726;
    border: 1px solid #2f3852;
    padding: 8px 14px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected {
    background: #243151;
    color: #ffffff;
}
QSlider::groove:horizontal {
    border: 1px solid #3d4f78;
    height: 6px;
    background: #101522;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #8ce7ff;
    border: 1px solid #b5f3ff;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QProgressBar {
    background-color: #0f1523;
    border: 1px solid #32446c;
    border-radius: 8px;
    color: #d9ecff;
    text-align: center;
    min-height: 22px;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #03ffd5,
        stop:0.5 #2f8cff,
        stop:1 #7a5cff
    );
}
QPlainTextEdit {
    background-color: #0b1019;
    border: 1px solid #2d3958;
    border-radius: 10px;
    color: #dbe8ff;
    font-family: "Consolas", "JetBrains Mono", monospace;
    font-size: 12px;
    padding: 6px;
}
QLabel#previewBox {
    background: #0d1119;
    border: 1px solid #2f3852;
    border-radius: 10px;
    color: #9db1da;
}
QSplitter::handle {
    background: #2b3348;
}
)";

QString formatSliderLabel(const QString &name, int value, const QString &suffix = QString())
{
    return QStringLiteral("%1: %2%3").arg(name).arg(value).arg(suffix);
}

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
        setWindowTitle(QStringLiteral("SeedVR2 Professional Upscaler (Qt6)"));
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

    void browseModelPath()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Model Path"),
            m_modelPath->text(),
            QStringLiteral("Models (*.safetensors *.pt *.pth *.onnx *.gguf);;All files (*)"));
        if (!path.isEmpty()) {
            m_modelPath->setText(path);
            m_modelPreset->setCurrentText(QStringLiteral("Custom"));
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

    void updateSliderLabels()
    {
        m_grainValue->setText(formatSliderLabel(QStringLiteral("Grain Amount"), m_grainSlider->value()));
        m_recoverValue->setText(formatSliderLabel(QStringLiteral("Recover Detail"), m_recoverSlider->value()));
    }

    void applyModelPreset(const QString &preset)
    {
        const QString current = m_modelPath->text().trimmed();
        if (!current.isEmpty() && QFileInfo::exists(current)) {
            return;
        }

        if (preset == QStringLiteral("3B")) {
            m_modelPath->setPlaceholderText(QStringLiteral("models/SEEDVR2/seedvr2_ema_3b.safetensors"));
        } else if (preset == QStringLiteral("7B")) {
            m_modelPath->setPlaceholderText(QStringLiteral("models/SEEDVR2/seedvr2_ema_7b.safetensors"));
        } else if (preset == QStringLiteral("GGUF")) {
            m_modelPath->setPlaceholderText(QStringLiteral("models/SEEDVR2/seedvr2_ema_7b-Q4_K_M.gguf"));
        } else {
            m_modelPath->setPlaceholderText(QStringLiteral("Custom model path"));
        }
    }

    void startRender()
    {
        const QString input = m_inputPath->text().trimmed();
        const QString output = m_outputPath->text().trimmed();
        const QString model = m_modelPath->text().trimmed();

        if (input.isEmpty() || output.isEmpty() || model.isEmpty()) {
            QMessageBox::warning(this,
                                 QStringLiteral("Missing Fields"),
                                 QStringLiteral("Please provide Input, Output, and Model path before starting."));
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
        args << QStringLiteral("--input") << input
             << QStringLiteral("--output") << output
             << QStringLiteral("--model") << model
             << QStringLiteral("--model-preset") << m_modelPreset->currentText().toLower()
             << QStringLiteral("--grain") << QString::number(m_grainSlider->value())
             << QStringLiteral("--recover-detail") << QString::number(m_recoverSlider->value())
             << QStringLiteral("--fps") << QString::number(m_fpsSpin->value())
             << QStringLiteral("--seed") << QString::number(m_seedSpin->value())
             << QStringLiteral("--attention-mode") << m_attentionMode->currentText()
             << QStringLiteral("--device") << m_deviceCombo->currentText()
             << QStringLiteral("--tile-size") << QString::number(m_tileSizeSpin->value())
             << QStringLiteral("--flush-interval") << QString::number(m_flushIntervalSpin->value());

        if (!m_previewInputPath->text().trimmed().isEmpty()) {
            args << QStringLiteral("--preview-input") << m_previewInputPath->text().trimmed();
        }
        if (!m_previewOutputPath->text().trimmed().isEmpty()) {
            args << QStringLiteral("--preview-output") << m_previewOutputPath->text().trimmed();
        }
        if (m_verboseDebug->isChecked()) {
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

        for (QWidget *widget : {
                 static_cast<QWidget *>(m_inputPath),
                 static_cast<QWidget *>(m_outputPath),
                 static_cast<QWidget *>(m_modelPath),
                 static_cast<QWidget *>(m_pythonPath),
                 static_cast<QWidget *>(m_inputBrowseButton),
                 static_cast<QWidget *>(m_outputBrowseButton),
                 static_cast<QWidget *>(m_modelBrowseButton),
                 static_cast<QWidget *>(m_pythonBrowseButton),
                 static_cast<QWidget *>(m_grainSlider),
                 static_cast<QWidget *>(m_recoverSlider),
                 static_cast<QWidget *>(m_fpsSpin),
                 static_cast<QWidget *>(m_seedSpin),
                 static_cast<QWidget *>(m_attentionMode),
                 static_cast<QWidget *>(m_deviceCombo),
                 static_cast<QWidget *>(m_tileSizeSpin),
                 static_cast<QWidget *>(m_flushIntervalSpin),
                 static_cast<QWidget *>(m_modelPreset),
                 static_cast<QWidget *>(m_verboseDebug) }) {
            if (widget) {
                widget->setEnabled(!running);
            }
        }
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

        auto *previewSubtitle = new QLabel(QStringLiteral("Multi-tab / split side-by-side comparison"), previewFrame);
        previewSubtitle->setObjectName(QStringLiteral("subtleTitle"));
        previewLayout->addWidget(previewSubtitle);

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

        auto *settingsTitle = new QLabel(QStringLiteral("Settings Panel"), rightFrame);
        settingsTitle->setObjectName(QStringLiteral("panelTitle"));
        rightLayout->addWidget(settingsTitle);

        auto *settingsTabs = new QTabWidget(rightFrame);

        auto *pathsTab = new QWidget(settingsTabs);
        auto *pathsForm = new QFormLayout(pathsTab);

        m_pythonPath = new QLineEdit(pathsTab);
        m_pythonBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsTab);
        pathsForm->addRow(QStringLiteral("Python Executable:"), buildPathRow(pathsTab, m_pythonPath, m_pythonBrowseButton));

        m_inputPath = new QLineEdit(pathsTab);
        m_inputBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsTab);
        pathsForm->addRow(QStringLiteral("Input Path:"), buildPathRow(pathsTab, m_inputPath, m_inputBrowseButton));

        m_outputPath = new QLineEdit(pathsTab);
        m_outputBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsTab);
        pathsForm->addRow(QStringLiteral("Output Path:"), buildPathRow(pathsTab, m_outputPath, m_outputBrowseButton));

        m_modelPath = new QLineEdit(pathsTab);
        m_modelBrowseButton = new QPushButton(QStringLiteral("Browse"), pathsTab);
        pathsForm->addRow(QStringLiteral("Model File:"), buildPathRow(pathsTab, m_modelPath, m_modelBrowseButton));

        settingsTabs->addTab(pathsTab, QStringLiteral("Paths"));

        auto *aiTab = new QWidget(settingsTabs);
        auto *aiLayout = new QVBoxLayout(aiTab);

        auto *aiForm = new QFormLayout();
        m_modelPreset = new QComboBox(aiTab);
        m_modelPreset->addItems({ QStringLiteral("3B"), QStringLiteral("7B"), QStringLiteral("GGUF"), QStringLiteral("Custom") });
        aiForm->addRow(QStringLiteral("AI Model Option:"), m_modelPreset);

        m_attentionMode = new QComboBox(aiTab);
        m_attentionMode->addItems({ QStringLiteral("sageattn_3"), QStringLiteral("flash_attn_3"), QStringLiteral("flash_attn_2"), QStringLiteral("sdpa") });
        aiForm->addRow(QStringLiteral("Attention Mode:"), m_attentionMode);

        m_deviceCombo = new QComboBox(aiTab);
        m_deviceCombo->addItems({ QStringLiteral("auto"), QStringLiteral("cuda"), QStringLiteral("cpu") });
        aiForm->addRow(QStringLiteral("Device:"), m_deviceCombo);

        m_seedSpin = new QSpinBox(aiTab);
        m_seedSpin->setRange(0, 2147483647);
        m_seedSpin->setValue(313);
        aiForm->addRow(QStringLiteral("Seed:"), m_seedSpin);

        aiLayout->addLayout(aiForm);

        auto *grainGroup = new QGroupBox(QStringLiteral("Image Controls"), aiTab);
        auto *grainLayout = new QVBoxLayout(grainGroup);

        m_grainValue = new QLabel(grainGroup);
        m_grainSlider = new QSlider(Qt::Horizontal, grainGroup);
        m_grainSlider->setRange(0, 100);
        m_grainSlider->setValue(12);

        m_recoverValue = new QLabel(grainGroup);
        m_recoverSlider = new QSlider(Qt::Horizontal, grainGroup);
        m_recoverSlider->setRange(0, 100);
        m_recoverSlider->setValue(35);

        grainLayout->addWidget(m_grainValue);
        grainLayout->addWidget(m_grainSlider);
        grainLayout->addWidget(m_recoverValue);
        grainLayout->addWidget(m_recoverSlider);
        aiLayout->addWidget(grainGroup);
        aiLayout->addStretch(1);

        settingsTabs->addTab(aiTab, QStringLiteral("AI"));

        auto *runtimeTab = new QWidget(settingsTabs);
        auto *runtimeForm = new QFormLayout(runtimeTab);

        m_fpsSpin = new QSpinBox(runtimeTab);
        m_fpsSpin->setRange(1, 240);
        m_fpsSpin->setValue(24);
        runtimeForm->addRow(QStringLiteral("FPS:"), m_fpsSpin);

        m_tileSizeSpin = new QSpinBox(runtimeTab);
        m_tileSizeSpin->setRange(64, 4096);
        m_tileSizeSpin->setValue(1024);
        runtimeForm->addRow(QStringLiteral("Tile Size:"), m_tileSizeSpin);

        m_flushIntervalSpin = new QSpinBox(runtimeTab);
        m_flushIntervalSpin->setRange(1, 512);
        m_flushIntervalSpin->setValue(8);
        runtimeForm->addRow(QStringLiteral("Memory Flush Interval (frames):"), m_flushIntervalSpin);

        m_verboseDebug = new QCheckBox(runtimeTab);
        runtimeForm->addRow(QStringLiteral("Verbose Debug:"), m_verboseDebug);

        settingsTabs->addTab(runtimeTab, QStringLiteral("Runtime"));

        rightLayout->addWidget(settingsTabs, 1);

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
        connect(m_modelBrowseButton, &QPushButton::clicked, this, &MainWindow::browseModelPath);
        connect(m_pythonBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePythonExecutable);
        connect(m_previewInputBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePreviewInput);
        connect(m_previewOutputBrowseButton, &QPushButton::clicked, this, &MainWindow::browsePreviewOutput);
        connect(previewRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshPreviewPanels);

        connect(m_grainSlider, &QSlider::valueChanged, this, &MainWindow::updateSliderLabels);
        connect(m_recoverSlider, &QSlider::valueChanged, this, &MainWindow::updateSliderLabels);
        connect(m_modelPreset, &QComboBox::currentTextChanged, this, &MainWindow::applyModelPreset);

        connect(m_startButton, &QPushButton::clicked, this, &MainWindow::startRender);
        connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::stopRender);

        updateSliderLabels();
        refreshPreviewPanels();
        statusBar()->showMessage(QStringLiteral("Ready"));
    }

    void saveSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        settings.setValue(QStringLiteral("inputPath"), m_inputPath->text());
        settings.setValue(QStringLiteral("outputPath"), m_outputPath->text());
        settings.setValue(QStringLiteral("modelPath"), m_modelPath->text());
        settings.setValue(QStringLiteral("pythonPath"), m_pythonPath->text());
        settings.setValue(QStringLiteral("previewInputPath"), m_previewInputPath->text());
        settings.setValue(QStringLiteral("previewOutputPath"), m_previewOutputPath->text());
        settings.setValue(QStringLiteral("modelPreset"), m_modelPreset->currentText());
        settings.setValue(QStringLiteral("attentionMode"), m_attentionMode->currentText());
        settings.setValue(QStringLiteral("device"), m_deviceCombo->currentText());
        settings.setValue(QStringLiteral("grain"), m_grainSlider->value());
        settings.setValue(QStringLiteral("recover"), m_recoverSlider->value());
        settings.setValue(QStringLiteral("fps"), m_fpsSpin->value());
        settings.setValue(QStringLiteral("seed"), m_seedSpin->value());
        settings.setValue(QStringLiteral("tileSize"), m_tileSizeSpin->value());
        settings.setValue(QStringLiteral("flushInterval"), m_flushIntervalSpin->value());
        settings.setValue(QStringLiteral("verboseDebug"), m_verboseDebug->isChecked());
    }

    void loadSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        m_inputPath->setText(settings.value(QStringLiteral("inputPath")).toString());
        m_outputPath->setText(settings.value(QStringLiteral("outputPath")).toString());
        m_modelPath->setText(settings.value(QStringLiteral("modelPath")).toString());
        m_pythonPath->setText(settings.value(QStringLiteral("pythonPath"), QStringLiteral("python")).toString());
        m_previewInputPath->setText(settings.value(QStringLiteral("previewInputPath")).toString());
        m_previewOutputPath->setText(settings.value(QStringLiteral("previewOutputPath")).toString());

        const QString modelPreset = settings.value(QStringLiteral("modelPreset"), QStringLiteral("3B")).toString();
        const int modelPresetIndex = m_modelPreset->findText(modelPreset);
        if (modelPresetIndex >= 0) {
            m_modelPreset->setCurrentIndex(modelPresetIndex);
        }

        const QString attentionMode = settings.value(QStringLiteral("attentionMode"), QStringLiteral("sageattn_3")).toString();
        const int attentionIndex = m_attentionMode->findText(attentionMode);
        if (attentionIndex >= 0) {
            m_attentionMode->setCurrentIndex(attentionIndex);
        }

        const QString device = settings.value(QStringLiteral("device"), QStringLiteral("auto")).toString();
        const int deviceIndex = m_deviceCombo->findText(device);
        if (deviceIndex >= 0) {
            m_deviceCombo->setCurrentIndex(deviceIndex);
        }

        m_grainSlider->setValue(settings.value(QStringLiteral("grain"), 12).toInt());
        m_recoverSlider->setValue(settings.value(QStringLiteral("recover"), 35).toInt());
        m_fpsSpin->setValue(settings.value(QStringLiteral("fps"), 24).toInt());
        m_seedSpin->setValue(settings.value(QStringLiteral("seed"), 313).toInt());
        m_tileSizeSpin->setValue(settings.value(QStringLiteral("tileSize"), 1024).toInt());
        m_flushIntervalSpin->setValue(settings.value(QStringLiteral("flushInterval"), 8).toInt());
        m_verboseDebug->setChecked(settings.value(QStringLiteral("verboseDebug"), false).toBool());

        updateSliderLabels();
        refreshPreviewPanels();
    }

    QLineEdit *m_inputPath = nullptr;
    QLineEdit *m_outputPath = nullptr;
    QLineEdit *m_modelPath = nullptr;
    QLineEdit *m_pythonPath = nullptr;

    QLineEdit *m_previewInputPath = nullptr;
    QLineEdit *m_previewOutputPath = nullptr;

    QPushButton *m_inputBrowseButton = nullptr;
    QPushButton *m_outputBrowseButton = nullptr;
    QPushButton *m_modelBrowseButton = nullptr;
    QPushButton *m_pythonBrowseButton = nullptr;
    QPushButton *m_previewInputBrowseButton = nullptr;
    QPushButton *m_previewOutputBrowseButton = nullptr;

    QLabel *m_splitInputPreview = nullptr;
    QLabel *m_splitOutputPreview = nullptr;
    QLabel *m_inputOnlyPreview = nullptr;
    QLabel *m_outputOnlyPreview = nullptr;
    QTabWidget *m_previewTabs = nullptr;

    QComboBox *m_modelPreset = nullptr;
    QComboBox *m_attentionMode = nullptr;
    QComboBox *m_deviceCombo = nullptr;

    QSlider *m_grainSlider = nullptr;
    QSlider *m_recoverSlider = nullptr;
    QLabel *m_grainValue = nullptr;
    QLabel *m_recoverValue = nullptr;

    QSpinBox *m_fpsSpin = nullptr;
    QSpinBox *m_seedSpin = nullptr;
    QSpinBox *m_tileSizeSpin = nullptr;
    QSpinBox *m_flushIntervalSpin = nullptr;
    QCheckBox *m_verboseDebug = nullptr;

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
