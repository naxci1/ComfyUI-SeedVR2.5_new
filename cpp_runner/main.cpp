#include "ProcessEngine.h"

#include <QApplication>
#include <QCloseEvent>
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
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSettings>
#include <QSlider>
#include <QSplitter>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QPixmap>

namespace {

constexpr const char *kStyleSheet = R"(
QWidget {
    background-color: #171a22;
    color: #e6ebff;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QMainWindow {
    background-color: #11141c;
}
QFrame#previewPanel, QFrame#controlsPanel, QFrame#logPanel {
    background-color: #202433;
    border: 1px solid #2f3550;
    border-radius: 12px;
}
QLabel#panelTitle {
    font-size: 16px;
    font-weight: 700;
    color: #eaf1ff;
}
QLineEdit {
    background-color: #151a27;
    border: 1px solid #3a4463;
    border-radius: 8px;
    padding: 6px 8px;
}
QPushButton {
    background-color: #2b3552;
    border: 1px solid #435181;
    border-radius: 8px;
    padding: 6px 12px;
    color: #e6ebff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #344166;
}
QPushButton#startButton {
    background-color: #0f5fff;
    border: none;
    color: #ffffff;
    font-size: 14px;
}
QPushButton#stopButton {
    background-color: #7a1f2f;
    border: none;
    color: #ffffff;
    font-size: 14px;
}
QSlider::groove:horizontal {
    border: 1px solid #364468;
    height: 6px;
    background: #151a27;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #9fe2ff;
    border: 1px solid #b4f1ff;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QProgressBar {
    background-color: #121724;
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
    background-color: #0d1119;
    border: 1px solid #2d3958;
    border-radius: 10px;
    color: #dbe8ff;
    font-family: "Consolas", "JetBrains Mono", monospace;
    font-size: 12px;
    padding: 6px;
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

} // namespace

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr)
        : QMainWindow(parent)
        , m_engine(new ProcessEngine(this))
    {
        setWindowTitle(QStringLiteral("SeedVR2 Professional Video Upscaler"));
        setMinimumSize(1366, 860);

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
    void browseInputDirectory()
    {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Input Directory"), m_inputDir->text());
        if (!dir.isEmpty()) {
            m_inputDir->setText(dir);
        }
    }

    void browseOutputDirectory()
    {
        const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), m_outputDir->text());
        if (!dir.isEmpty()) {
            m_outputDir->setText(dir);
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
        }
    }

    void updateSliderLabels()
    {
        m_grainValue->setText(formatSliderLabel(QStringLiteral("Grain Amount"), m_grainSlider->value()));
        m_recoverValue->setText(formatSliderLabel(QStringLiteral("Recover Detail"), m_recoverSlider->value()));
        m_fpsValue->setText(formatSliderLabel(QStringLiteral("FPS"), m_fpsSlider->value()));
    }

    void startRender()
    {
        const QString input = m_inputDir->text().trimmed();
        const QString output = m_outputDir->text().trimmed();
        const QString model = m_modelPath->text().trimmed();

        if (input.isEmpty() || output.isEmpty() || model.isEmpty()) {
            QMessageBox::warning(
                this,
                QStringLiteral("Missing Fields"),
                QStringLiteral("Please provide Input Directory, Output Directory, and Model Path before starting."));
            return;
        }

        const QString pythonExe = m_pythonExecutable;
        const QString scriptPath = findDefaultCliScriptPath();
        if (!QFileInfo::exists(scriptPath)) {
            QMessageBox::critical(
                this,
                QStringLiteral("CLI Not Found"),
                QStringLiteral("Could not locate inference_cli.py. Expected near executable or repository root."));
            return;
        }

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
             << QStringLiteral("--grain") << QString::number(m_grainSlider->value())
             << QStringLiteral("--recover-detail") << QString::number(m_recoverSlider->value())
             << QStringLiteral("--fps") << QString::number(m_fpsSlider->value());

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
        m_batchLabel->setText(
            QStringLiteral("Batch Progress: %1/%2")
                .arg(doneFiles)
                .arg(totalFiles));
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
        }
    }

private:
    void setRunning(bool running)
    {
        m_startButton->setEnabled(!running);
        m_stopButton->setEnabled(running);
        m_inputDir->setEnabled(!running);
        m_outputDir->setEnabled(!running);
        m_modelPath->setEnabled(!running);
        m_inputBrowseButton->setEnabled(!running);
        m_outputBrowseButton->setEnabled(!running);
        m_modelBrowseButton->setEnabled(!running);
        m_grainSlider->setEnabled(!running);
        m_recoverSlider->setEnabled(!running);
        m_fpsSlider->setEnabled(!running);
    }

    void buildUi()
    {
        auto *central = new QWidget(this);
        setCentralWidget(central);

        auto *rootLayout = new QVBoxLayout(central);
        rootLayout->setContentsMargins(12, 12, 12, 12);
        rootLayout->setSpacing(10);

        auto *mainVerticalSplitter = new QSplitter(Qt::Vertical, central);

        auto *topHorizontalSplitter = new QSplitter(Qt::Horizontal, mainVerticalSplitter);

        auto *previewPanel = new QFrame(topHorizontalSplitter);
        previewPanel->setObjectName(QStringLiteral("previewPanel"));
        auto *previewLayout = new QVBoxLayout(previewPanel);
        previewLayout->setContentsMargins(12, 12, 12, 12);
        previewLayout->setSpacing(10);

        auto *previewTitle = new QLabel(QStringLiteral("Frame Preview"), previewPanel);
        previewTitle->setObjectName(QStringLiteral("panelTitle"));
        previewLayout->addWidget(previewTitle);

        m_previewLabel = new QLabel(QStringLiteral("No frame available"), previewPanel);
        m_previewLabel->setAlignment(Qt::AlignCenter);
        m_previewLabel->setMinimumSize(760, 480);
        m_previewLabel->setStyleSheet(QStringLiteral("background:#0f131d;border:1px solid #2b3550;border-radius:10px;"));
        previewLayout->addWidget(m_previewLabel, 1);

        auto *controlsPanel = new QFrame(topHorizontalSplitter);
        controlsPanel->setObjectName(QStringLiteral("controlsPanel"));
        auto *controlsLayout = new QVBoxLayout(controlsPanel);
        controlsLayout->setContentsMargins(12, 12, 12, 12);
        controlsLayout->setSpacing(10);

        auto *controlsTitle = new QLabel(QStringLiteral("Control Panel"), controlsPanel);
        controlsTitle->setObjectName(QStringLiteral("panelTitle"));
        controlsLayout->addWidget(controlsTitle);

        auto *formGroup = new QGroupBox(QStringLiteral("Paths"), controlsPanel);
        auto *formLayout = new QFormLayout(formGroup);
        formLayout->setSpacing(8);

        m_inputDir = new QLineEdit(formGroup);
        m_inputDir->setPlaceholderText(QStringLiteral("Input directory with videos/images"));
        m_inputBrowseButton = new QPushButton(QStringLiteral("Browse"), formGroup);
        formLayout->addRow(QStringLiteral("Input:"), buildPathRow(formGroup, m_inputDir, m_inputBrowseButton));

        m_outputDir = new QLineEdit(formGroup);
        m_outputDir->setPlaceholderText(QStringLiteral("Output directory"));
        m_outputBrowseButton = new QPushButton(QStringLiteral("Browse"), formGroup);
        formLayout->addRow(QStringLiteral("Output:"), buildPathRow(formGroup, m_outputDir, m_outputBrowseButton));

        m_modelPath = new QLineEdit(formGroup);
        m_modelPath->setPlaceholderText(QStringLiteral("Model path"));
        m_modelBrowseButton = new QPushButton(QStringLiteral("Browse"), formGroup);
        formLayout->addRow(QStringLiteral("Model:"), buildPathRow(formGroup, m_modelPath, m_modelBrowseButton));

        controlsLayout->addWidget(formGroup);

        auto *sliderGroup = new QGroupBox(QStringLiteral("Adjustments"), controlsPanel);
        auto *sliderLayout = new QVBoxLayout(sliderGroup);

        m_grainValue = new QLabel(sliderGroup);
        m_grainSlider = new QSlider(Qt::Horizontal, sliderGroup);
        m_grainSlider->setRange(0, 100);
        m_grainSlider->setValue(12);
        sliderLayout->addWidget(m_grainValue);
        sliderLayout->addWidget(m_grainSlider);

        m_recoverValue = new QLabel(sliderGroup);
        m_recoverSlider = new QSlider(Qt::Horizontal, sliderGroup);
        m_recoverSlider->setRange(0, 100);
        m_recoverSlider->setValue(35);
        sliderLayout->addWidget(m_recoverValue);
        sliderLayout->addWidget(m_recoverSlider);

        m_fpsValue = new QLabel(sliderGroup);
        m_fpsSlider = new QSlider(Qt::Horizontal, sliderGroup);
        m_fpsSlider->setRange(1, 120);
        m_fpsSlider->setValue(24);
        sliderLayout->addWidget(m_fpsValue);
        sliderLayout->addWidget(m_fpsSlider);

        controlsLayout->addWidget(sliderGroup);

        auto *buttonRow = new QHBoxLayout();
        m_startButton = new QPushButton(QStringLiteral("Start Render"), controlsPanel);
        m_startButton->setObjectName(QStringLiteral("startButton"));
        m_stopButton = new QPushButton(QStringLiteral("Stop"), controlsPanel);
        m_stopButton->setObjectName(QStringLiteral("stopButton"));
        buttonRow->addWidget(m_startButton);
        buttonRow->addWidget(m_stopButton);
        controlsLayout->addLayout(buttonRow);

        controlsLayout->addStretch();

        topHorizontalSplitter->addWidget(previewPanel);
        topHorizontalSplitter->addWidget(controlsPanel);
        topHorizontalSplitter->setStretchFactor(0, 4);
        topHorizontalSplitter->setStretchFactor(1, 2);

        auto *bottomContainer = new QWidget(mainVerticalSplitter);
        auto *bottomLayout = new QVBoxLayout(bottomContainer);
        bottomLayout->setContentsMargins(0, 0, 0, 0);
        bottomLayout->setSpacing(10);

        auto *progressGroup = new QGroupBox(QStringLiteral("Progress"), bottomContainer);
        auto *progressLayout = new QVBoxLayout(progressGroup);
        progressLayout->setSpacing(6);

        m_currentFileLabel = new QLabel(QStringLiteral("Current File: — (0/0)"), progressGroup);
        m_fileProgressBar = new QProgressBar(progressGroup);
        m_fileProgressBar->setRange(0, 100);
        m_fileProgressBar->setValue(0);

        m_batchLabel = new QLabel(QStringLiteral("Batch Progress: 0/0"), progressGroup);
        m_batchProgressBar = new QProgressBar(progressGroup);
        m_batchProgressBar->setRange(0, 100);
        m_batchProgressBar->setValue(0);

        progressLayout->addWidget(m_currentFileLabel);
        progressLayout->addWidget(m_fileProgressBar);
        progressLayout->addWidget(m_batchLabel);
        progressLayout->addWidget(m_batchProgressBar);

        bottomLayout->addWidget(progressGroup);

        auto *logPanel = new QFrame(bottomContainer);
        logPanel->setObjectName(QStringLiteral("logPanel"));
        auto *logLayout = new QVBoxLayout(logPanel);
        logLayout->setContentsMargins(12, 12, 12, 12);
        logLayout->setSpacing(8);

        auto *logTitle = new QLabel(QStringLiteral("Log"), logPanel);
        logTitle->setObjectName(QStringLiteral("panelTitle"));
        m_logConsole = new QPlainTextEdit(logPanel);
        m_logConsole->setReadOnly(true);
        m_logConsole->setMaximumBlockCount(8000);
        m_logConsole->setMinimumHeight(240);

        logLayout->addWidget(logTitle);
        logLayout->addWidget(m_logConsole, 1);

        bottomLayout->addWidget(logPanel, 1);

        mainVerticalSplitter->addWidget(topHorizontalSplitter);
        mainVerticalSplitter->addWidget(bottomContainer);
        mainVerticalSplitter->setStretchFactor(0, 3);
        mainVerticalSplitter->setStretchFactor(1, 2);

        rootLayout->addWidget(mainVerticalSplitter);

        connect(m_inputBrowseButton, &QPushButton::clicked, this, &MainWindow::browseInputDirectory);
        connect(m_outputBrowseButton, &QPushButton::clicked, this, &MainWindow::browseOutputDirectory);
        connect(m_modelBrowseButton, &QPushButton::clicked, this, &MainWindow::browseModelPath);
        connect(m_grainSlider, &QSlider::valueChanged, this, &MainWindow::updateSliderLabels);
        connect(m_recoverSlider, &QSlider::valueChanged, this, &MainWindow::updateSliderLabels);
        connect(m_fpsSlider, &QSlider::valueChanged, this, &MainWindow::updateSliderLabels);
        connect(m_startButton, &QPushButton::clicked, this, &MainWindow::startRender);
        connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::stopRender);

        updateSliderLabels();
        statusBar()->showMessage(QStringLiteral("Ready"));
    }

    void saveSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        settings.setValue(QStringLiteral("inputDir"), m_inputDir->text());
        settings.setValue(QStringLiteral("outputDir"), m_outputDir->text());
        settings.setValue(QStringLiteral("modelPath"), m_modelPath->text());
        settings.setValue(QStringLiteral("grain"), m_grainSlider->value());
        settings.setValue(QStringLiteral("recover"), m_recoverSlider->value());
        settings.setValue(QStringLiteral("fps"), m_fpsSlider->value());
    }

    void loadSettings()
    {
        QSettings settings(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        m_inputDir->setText(settings.value(QStringLiteral("inputDir")).toString());
        m_outputDir->setText(settings.value(QStringLiteral("outputDir")).toString());
        m_modelPath->setText(settings.value(QStringLiteral("modelPath")).toString());
        m_grainSlider->setValue(settings.value(QStringLiteral("grain"), 12).toInt());
        m_recoverSlider->setValue(settings.value(QStringLiteral("recover"), 35).toInt());
        m_fpsSlider->setValue(settings.value(QStringLiteral("fps"), 24).toInt());
        updateSliderLabels();
    }

    QLineEdit *m_inputDir = nullptr;
    QLineEdit *m_outputDir = nullptr;
    QLineEdit *m_modelPath = nullptr;

    QPushButton *m_inputBrowseButton = nullptr;
    QPushButton *m_outputBrowseButton = nullptr;
    QPushButton *m_modelBrowseButton = nullptr;

    QSlider *m_grainSlider = nullptr;
    QSlider *m_recoverSlider = nullptr;
    QSlider *m_fpsSlider = nullptr;

    QLabel *m_grainValue = nullptr;
    QLabel *m_recoverValue = nullptr;
    QLabel *m_fpsValue = nullptr;

    QLabel *m_previewLabel = nullptr;

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
