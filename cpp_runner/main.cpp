/**
 * main.cpp – SeedVR2 C++ Runner
 *
 * Standalone Qt6 Widgets application that acts as a Topaz-style "runner.exe":
 *   • Dark-mode charcoal UI (#1A1A1A, border-radius: 10px)
 *   • Launches inference_cli.py through QProcess (non-blocking)
 *   • Dual QProgressBar with HH:MM:SS time estimates
 *   • Real-time async stdout scraping via ProcessEngine
 *
 * Build:  See CMakeLists.txt
 * Target: Windows 10/11, Qt 6.5+, MSVC 2022 or MinGW-w64
 */

#include "ProcessEngine.h"

#include <QApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QDoubleSpinBox>
#include <QMessageBox>
#include <QPushButton>
#include <QProgressBar>
#include <QScrollBar>
#include <QSettings>
#include <QSizePolicy>
#include <QStatusBar>
#include <QTextEdit>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>
#include <QIcon>
#include <QProcessEnvironment>

// ─────────────────────────────────────────────────────────────────────────────
// Dark-mode QSS stylesheet
// ─────────────────────────────────────────────────────────────────────────────

static const char *k_stylesheet = R"(
QWidget {
    background-color: #1A1A1A;
    color: #E3E4E6;
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1A1A1A;
}

QGroupBox {
    background-color: #222326;
    border: 1px solid #2F3136;
    border-radius: 10px;
    margin-top: 16px;
    padding: 10px 12px;
    font-weight: 600;
    color: #BFD2F5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -2px;
    padding: 0 4px;
    color: #BFD2F5;
}

QLineEdit, QDoubleSpinBox {
    background-color: #2A2D32;
    border: 1px solid #3A3E45;
    border-radius: 6px;
    padding: 5px 8px;
    color: #E3E4E6;
    selection-background-color: #0052CC;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #0052CC;
}

QPushButton {
    background-color: #2F3136;
    border: 1px solid #3F434A;
    border-radius: 8px;
    padding: 6px 14px;
    color: #E3E4E6;
    font-weight: 500;
}
QPushButton:hover  { background-color: #3A3E45; border-color: #505660; }
QPushButton:pressed { background-color: #252830; }

QPushButton#startBtn {
    background-color: #0052CC;
    border: none;
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 700;
    border-radius: 10px;
    padding: 8px 24px;
}
QPushButton#startBtn:hover  { background-color: #0066FF; }
QPushButton#startBtn:pressed { background-color: #003D99; }
QPushButton#startBtn:disabled { background-color: #334466; color: #668ABF; }

QPushButton#stopBtn {
    background-color: #8B0000;
    border: none;
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 700;
    border-radius: 10px;
    padding: 8px 24px;
}
QPushButton#stopBtn:hover  { background-color: #CC0000; }
QPushButton#stopBtn:pressed { background-color: #660000; }
QPushButton#stopBtn:disabled { background-color: #3D1111; color: #7A4444; }

QProgressBar {
    background-color: #2A2D32;
    border: 1px solid #3A3E45;
    border-radius: 8px;
    text-align: center;
    color: #E3E4E6;
    font-size: 11px;
    height: 22px;
}
QProgressBar::chunk {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #0044BB, stop:1 #0088FF);
    border-radius: 7px;
}

QLabel#progressLabel {
    color: #A8B8D8;
    font-size: 11px;
    padding: 2px 0;
}

QTextEdit {
    background-color: #141618;
    border: 1px solid #2A2D32;
    border-radius: 8px;
    color: #C8CDD6;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 4px;
}

QStatusBar {
    background-color: #111214;
    color: #888E99;
    border-top: 1px solid #2A2D32;
}

QScrollBar:vertical {
    background: #1E2126;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #404550;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #5A6070; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
)";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

QString formatHMS(double seconds)
{
    if (seconds < 0.0) seconds = 0.0;
    const int total = static_cast<int>(seconds);
    return QStringLiteral("%1:%2:%3")
        .arg(total / 3600,          2, 10, QLatin1Char('0'))
        .arg((total % 3600) / 60,   2, 10, QLatin1Char('0'))
        .arg(total % 60,            2, 10, QLatin1Char('0'));
}

QWidget *makeBrowseRow(QLineEdit *edit, const QString &dialogTitle, bool directory, QWidget *parent)
{
    auto *row    = new QWidget(parent);
    auto *layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(edit);

    auto *btn = new QPushButton(QStringLiteral("Browse…"), parent);
    btn->setFixedWidth(80);
    layout->addWidget(btn);

    QObject::connect(btn, &QPushButton::clicked, parent, [edit, dialogTitle, directory, parent]() {
        QString path;
        if (directory)
            path = QFileDialog::getExistingDirectory(parent, dialogTitle, edit->text());
        else
            path = QFileDialog::getOpenFileName(parent, dialogTitle, edit->text(),
                                                QStringLiteral("SafeTensors (*.safetensors *.gguf);;All files (*)"));
        if (!path.isEmpty())
            edit->setText(path);
    });

    return row;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// MainWindow
// ─────────────────────────────────────────────────────────────────────────────

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onStartClicked();
    void onStopClicked();
    void onLogLine(const QString &line);
    void onFileProgress(const QString &filename, int current, int total,
                        int doneFiles, int remainingFiles);
    void onBatchProgress(int current, int total);
    void onFinished(bool success, const QString &message);

private:
    void buildUi();
    void saveSettings();
    void loadSettings();
    void setRunning(bool running);

    // Settings / path inputs
    QLineEdit       *m_inputDir  = nullptr;
    QLineEdit       *m_outputDir = nullptr;
    QLineEdit       *m_modelPath = nullptr;
    QLineEdit       *m_pythonExe = nullptr;
    QLineEdit       *m_cliScript = nullptr;
    QDoubleSpinBox  *m_fpsSpin   = nullptr;

    // Controls
    QPushButton *m_startBtn = nullptr;
    QPushButton *m_stopBtn  = nullptr;

    // Progress – current file
    QProgressBar *m_fileBar   = nullptr;
    QLabel       *m_fileLabel = nullptr;

    // Progress – queue
    QProgressBar *m_queueBar   = nullptr;
    QLabel       *m_queueLabel = nullptr;

    // Log
    QTextEdit *m_logView = nullptr;

    // State
    ProcessEngine *m_engine          = nullptr;
    double         m_fps             = 1.8;
    int            m_totalQueueFiles = 0;
    int            m_lastFileTotalFrames = 0;
};

// ── Constructor ───────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_engine(new ProcessEngine(this))
{
    setWindowTitle(QStringLiteral("SeedVR2 Runner"));
    setMinimumSize(820, 680);
    buildUi();
    loadSettings();

    connect(m_engine, &ProcessEngine::logLine,
            this, &MainWindow::onLogLine);
    connect(m_engine, &ProcessEngine::fileProgressUpdated,
            this, &MainWindow::onFileProgress);
    connect(m_engine, &ProcessEngine::batchProgressUpdated,
            this, &MainWindow::onBatchProgress);
    connect(m_engine, &ProcessEngine::processingFinished,
            this, &MainWindow::onFinished);

    setRunning(false);
}

MainWindow::~MainWindow() = default;

// ── UI construction ───────────────────────────────────────────────────────────

void MainWindow::buildUi()
{
    auto *central = new QWidget(this);
    setCentralWidget(central);

    auto *root = new QVBoxLayout(central);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(12);

    // ── Paths group ──────────────────────────────────────────────────────
    auto *pathGroup  = new QGroupBox(QStringLiteral("Paths"), central);
    auto *pathForm   = new QFormLayout(pathGroup);
    pathForm->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    pathForm->setSpacing(8);

    m_inputDir  = new QLineEdit(pathGroup);
    m_inputDir->setPlaceholderText(QStringLiteral("Directory containing input videos / images…"));
    pathForm->addRow(QStringLiteral("Input Directory:"),
                     makeBrowseRow(m_inputDir, QStringLiteral("Select Input Directory"), true, pathGroup));

    m_outputDir = new QLineEdit(pathGroup);
    m_outputDir->setPlaceholderText(QStringLiteral("Directory for upscaled output files…"));
    pathForm->addRow(QStringLiteral("Output Directory:"),
                     makeBrowseRow(m_outputDir, QStringLiteral("Select Output Directory"), true, pathGroup));

    m_modelPath = new QLineEdit(pathGroup);
    m_modelPath->setPlaceholderText(QStringLiteral("Path to .safetensors or .gguf model file…"));
    pathForm->addRow(QStringLiteral("Model Path:"),
                     makeBrowseRow(m_modelPath, QStringLiteral("Select Model File"), false, pathGroup));

    m_pythonExe = new QLineEdit(pathGroup);
    m_pythonExe->setPlaceholderText(QStringLiteral(R"(C:\ComfyUI\python_embeded\python.exe)"));
    pathForm->addRow(QStringLiteral("Python Executable:"),
                     makeBrowseRow(m_pythonExe, QStringLiteral("Select Python Executable"), false, pathGroup));

    m_cliScript = new QLineEdit(pathGroup);
    m_cliScript->setPlaceholderText(QStringLiteral(R"(C:\ComfyUI\custom_nodes\ComfyUI-SeedVR2.5\inference_cli.py)"));
    pathForm->addRow(QStringLiteral("inference_cli.py:"),
                     makeBrowseRow(m_cliScript, QStringLiteral("Select inference_cli.py"), false, pathGroup));

    // ── Processing options ───────────────────────────────────────────────
    auto *optGroup = new QGroupBox(QStringLiteral("Processing Options"), central);
    auto *optForm  = new QFormLayout(optGroup);
    optForm->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    optForm->setSpacing(8);

    m_fpsSpin = new QDoubleSpinBox(optGroup);
    m_fpsSpin->setRange(0.01, 100.0);
    m_fpsSpin->setDecimals(2);
    m_fpsSpin->setSingleStep(0.1);
    m_fpsSpin->setValue(1.8);
    m_fpsSpin->setSuffix(QStringLiteral("  fps (estimated processing speed)"));
    m_fpsSpin->setFixedWidth(280);
    optForm->addRow(QStringLiteral("Processing Speed:"), m_fpsSpin);

    root->addWidget(pathGroup);
    root->addWidget(optGroup);

    // ── Action buttons ───────────────────────────────────────────────────
    auto *btnRow = new QHBoxLayout();
    m_startBtn = new QPushButton(QStringLiteral("▶  Start Render"), central);
    m_startBtn->setObjectName(QStringLiteral("startBtn"));
    m_startBtn->setFixedHeight(38);

    m_stopBtn = new QPushButton(QStringLiteral("⏹  Stop"), central);
    m_stopBtn->setObjectName(QStringLiteral("stopBtn"));
    m_stopBtn->setFixedHeight(38);

    btnRow->addStretch();
    btnRow->addWidget(m_startBtn);
    btnRow->addWidget(m_stopBtn);
    btnRow->addStretch();
    root->addLayout(btnRow);

    connect(m_startBtn, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    connect(m_stopBtn,  &QPushButton::clicked, this, &MainWindow::onStopClicked);

    // ── Progress bars ────────────────────────────────────────────────────
    auto *progressGroup = new QGroupBox(QStringLiteral("Progress"), central);
    auto *pgLayout      = new QVBoxLayout(progressGroup);
    pgLayout->setSpacing(6);

    // ProgressBar 1 – current file
    m_fileLabel = new QLabel(QStringLiteral("Current File: — | Remaining: --:--:-- | Frames: 0/0"), progressGroup);
    m_fileLabel->setObjectName(QStringLiteral("progressLabel"));
    m_fileBar   = new QProgressBar(progressGroup);
    m_fileBar->setRange(0, 100);
    m_fileBar->setValue(0);
    m_fileBar->setTextVisible(false);
    m_fileBar->setFixedHeight(22);

    // ProgressBar 2 – overall queue
    m_queueLabel = new QLabel(
        QStringLiteral("Overall Batch Progress | Completed: 0/0 | Estimated Total Time Left: --:--:--"),
        progressGroup);
    m_queueLabel->setObjectName(QStringLiteral("progressLabel"));
    m_queueBar   = new QProgressBar(progressGroup);
    m_queueBar->setRange(0, 100);
    m_queueBar->setValue(0);
    m_queueBar->setTextVisible(false);
    m_queueBar->setFixedHeight(22);

    pgLayout->addWidget(m_fileLabel);
    pgLayout->addWidget(m_fileBar);
    pgLayout->addSpacing(4);
    pgLayout->addWidget(m_queueLabel);
    pgLayout->addWidget(m_queueBar);

    root->addWidget(progressGroup);

    // ── Log output ───────────────────────────────────────────────────────
    auto *logGroup  = new QGroupBox(QStringLiteral("Log"), central);
    auto *logLayout = new QVBoxLayout(logGroup);
    m_logView = new QTextEdit(logGroup);
    m_logView->setReadOnly(true);
    m_logView->setLineWrapMode(QTextEdit::NoWrap);
    logLayout->addWidget(m_logView);
    root->addWidget(logGroup, 1);   // stretch the log to fill remaining space

    // ── Status bar ───────────────────────────────────────────────────────
    statusBar()->showMessage(QStringLiteral("Ready."));
}

// ── Slot implementations ──────────────────────────────────────────────────────

void MainWindow::onStartClicked()
{
    const QString inputDir  = m_inputDir->text().trimmed();
    const QString outputDir = m_outputDir->text().trimmed();
    const QString modelPath = m_modelPath->text().trimmed();
    const QString pythonExe = m_pythonExe->text().trimmed();
    const QString cliScript = m_cliScript->text().trimmed();

    if (inputDir.isEmpty() || outputDir.isEmpty() || modelPath.isEmpty() ||
        pythonExe.isEmpty() || cliScript.isEmpty())
    {
        QMessageBox::warning(this, QStringLiteral("Missing Paths"),
                             QStringLiteral("Please fill in all path fields before starting."));
        return;
    }

    m_fps = m_fpsSpin->value();
    m_totalQueueFiles  = 0;
    m_lastFileTotalFrames = 0;

    // Reset progress bars
    m_fileBar->setValue(0);
    m_queueBar->setValue(0);
    m_fileLabel->setText(QStringLiteral("Current File: — | Remaining: --:--:-- | Frames: 0/0"));
    m_queueLabel->setText(QStringLiteral("Overall Batch Progress | Completed: 0/0 | Estimated Total Time Left: --:--:--"));

    m_logView->clear();
    saveSettings();
    setRunning(true);

    // Build CLI arguments for inference_cli.py
    QStringList args;
    args << inputDir
         << QStringLiteral("--output_dir")   << outputDir
         << QStringLiteral("--dit_model_path") << modelPath;

    statusBar()->showMessage(QStringLiteral("Running…"));
    m_engine->startProcess(pythonExe, cliScript, args);
}

void MainWindow::onStopClicked()
{
    m_engine->stopProcess();
    statusBar()->showMessage(QStringLiteral("Stopping…"));
}

void MainWindow::onLogLine(const QString &line)
{
    m_logView->append(line);
    // Auto-scroll to bottom
    QScrollBar *sb = m_logView->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void MainWindow::onFileProgress(const QString &filename,
                                int current, int total,
                                int doneFiles, int remainingFiles)
{
    m_lastFileTotalFrames = (total > 0) ? total : m_lastFileTotalFrames;

    // ── File progress bar ────────────────────────────────────────────────
    const int filePercent = (total > 0) ? qBound(0, current * 100 / total, 100) : 0;
    m_fileBar->setValue(filePercent);

    const double fileRemainSec = (m_fps > 0.0 && total > current)
                                 ? static_cast<double>(total - current) / m_fps
                                 : 0.0;

    const QString baseName = QFileInfo(filename).fileName();
    m_fileLabel->setText(
        QStringLiteral("Current File: %1 | Remaining: %2 | Frames: %3/%4")
            .arg(baseName.isEmpty() ? QStringLiteral("—") : baseName)
            .arg(formatHMS(fileRemainSec))
            .arg(current)
            .arg(total));

    // ── Queue progress bar ───────────────────────────────────────────────
    const int totalFiles = doneFiles + 1 + remainingFiles;  // done + current + remaining
    if (m_totalQueueFiles == 0 && totalFiles > 0)
        m_totalQueueFiles = totalFiles;

    const int queueTotal = qMax(m_totalQueueFiles, totalFiles);
    const int queuePercent = (queueTotal > 0)
                             ? qBound(0, doneFiles * 100 / queueTotal, 100)
                             : 0;
    m_queueBar->setValue(queuePercent);

    // Estimate total queue remaining: remaining files × current file frame count + frames left in current file
    const double queueRemainSec = (m_fps > 0.0)
        ? static_cast<double>(remainingFiles * m_lastFileTotalFrames + qMax(0, total - current)) / m_fps
        : 0.0;

    m_queueLabel->setText(
        QStringLiteral("Overall Batch Progress | Completed: %1/%2 | Estimated Total Time Left: %3")
            .arg(doneFiles)
            .arg(queueTotal)
            .arg(formatHMS(queueRemainSec)));
}

void MainWindow::onBatchProgress(int current, int total)
{
    // Inner diffusion-step bar reused on the file bar when no status token available.
    // This is the fallback for "step N/M" token parsing.
    if (total > 0) {
        const int pct = qBound(0, current * 100 / total, 100);
        // Only update if the file bar has no meaningful data yet
        if (m_fileBar->value() == 0)
            m_fileBar->setValue(pct);
    }
}

void MainWindow::onFinished(bool success, const QString &message)
{
    setRunning(false);
    if (success) {
        m_fileBar->setValue(100);
        m_queueBar->setValue(100);
        statusBar()->showMessage(QStringLiteral("✅  Completed: ") + message);
    } else {
        statusBar()->showMessage(QStringLiteral("⏹  ") + message);
    }
}

// ── State helpers ─────────────────────────────────────────────────────────────

void MainWindow::setRunning(bool running)
{
    m_startBtn->setEnabled(!running);
    m_stopBtn->setEnabled(running);
    m_inputDir->setEnabled(!running);
    m_outputDir->setEnabled(!running);
    m_modelPath->setEnabled(!running);
    m_pythonExe->setEnabled(!running);
    m_cliScript->setEnabled(!running);
    m_fpsSpin->setEnabled(!running);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    if (m_engine->isRunning()) {
        m_engine->stopProcess();
    }
    QMainWindow::closeEvent(event);
}

// ── Persistence (QSettings) ──────────────────────────────────────────────────

void MainWindow::saveSettings()
{
    QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("Runner"));
    s.setValue(QStringLiteral("inputDir"),  m_inputDir->text());
    s.setValue(QStringLiteral("outputDir"), m_outputDir->text());
    s.setValue(QStringLiteral("modelPath"), m_modelPath->text());
    s.setValue(QStringLiteral("pythonExe"), m_pythonExe->text());
    s.setValue(QStringLiteral("cliScript"), m_cliScript->text());
    s.setValue(QStringLiteral("fps"),       m_fpsSpin->value());
}

void MainWindow::loadSettings()
{
    QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("Runner"));
    m_inputDir->setText(s.value(QStringLiteral("inputDir")).toString());
    m_outputDir->setText(s.value(QStringLiteral("outputDir")).toString());
    m_modelPath->setText(s.value(QStringLiteral("modelPath")).toString());
    m_pythonExe->setText(s.value(QStringLiteral("pythonExe"),
        QStringLiteral(R"(C:\ComfyUI\python_embeded\python.exe)")).toString());
    m_cliScript->setText(s.value(QStringLiteral("cliScript")).toString());
    m_fpsSpin->setValue(s.value(QStringLiteral("fps"), 1.8).toDouble());
}

// ─────────────────────────────────────────────────────────────────────────────
// main()
// ─────────────────────────────────────────────────────────────────────────────

#include "main.moc"

int main(int argc, char *argv[])
{
    // High-DPI support (relevant on Windows with display scaling)
    QApplication::setHighDpiScaleFactorRoundingPolicy(
        Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("SeedVR2 Runner"));
    app.setOrganizationName(QStringLiteral("SeedVR2"));
    app.setApplicationVersion(QStringLiteral("2.5"));
    app.setStyleSheet(QString::fromLatin1(k_stylesheet));

    MainWindow window;
    window.show();

    return app.exec();
}
