#include "ProcessEngine.h"

#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
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
#include <QScrollArea>
#include <QSettings>
#include <QSlider>
#include <QStatusBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <QPixmap>
#include <QSignalBlocker>

namespace {

constexpr const char *kStyle = R"(
QWidget { background:#181a1f; color:#e7e9ee; font:13px "Segoe UI"; }
QFrame#previewFrame, QFrame#settingsFrame, QFrame#timelineFrame, QFrame#logFrame {
    background:#22252d; border:1px solid #2f3441; border-radius:10px;
}
QPushButton { background:#2f3441; border:1px solid #3c4353; border-radius:8px; padding:6px 12px; }
QPushButton:hover { background:#3a4253; }
QPushButton#startBtn { background:#1d6fff; border:none; font-weight:700; color:white; }
QPushButton#stopBtn { background:#7a1c1c; border:none; font-weight:700; color:white; }
QLineEdit, QComboBox, QToolButton { background:#1a1e26; border:1px solid #343b49; border-radius:6px; padding:5px 8px; }
QSlider::groove:horizontal { height:6px; background:#2f3441; border-radius:3px; }
QSlider::handle:horizontal { width:14px; background:#6ea8ff; margin:-4px 0; border-radius:7px; }
QProgressBar { background:#1a1e26; border:1px solid #343b49; border-radius:6px; text-align:center; height:20px; }
QProgressBar::chunk { background:#1d6fff; border-radius:6px; }
QPlainTextEdit { background:#11141a; border:1px solid #2a303d; border-radius:8px; font:12px "Consolas"; }
QLabel#thumbnailLabel { background:#0f1116; border:1px solid #343b49; border-radius:6px; padding:2px; }
)";

QStringList imageExtensions()
{
    return {QStringLiteral("*.png"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.bmp"), QStringLiteral("*.webp")};
}

QStringList collectImageFiles(const QString &inputPath)
{
    QFileInfo info(inputPath);
    if (!info.exists()) return {};

    if (info.isFile()) {
        return {info.absoluteFilePath()};
    }

    QDir dir(info.absoluteFilePath());
    QStringList files;
    for (const QString &pattern : imageExtensions()) {
        const QFileInfoList list = dir.entryInfoList({pattern}, QDir::Files, QDir::Name);
        for (const QFileInfo &entry : list) files.push_back(entry.absoluteFilePath());
    }
    files.removeDuplicates();
    files.sort();
    return files;
}

QString humanName(const QString &path)
{
    const QString name = QFileInfo(path).fileName();
    return name.isEmpty() ? QStringLiteral("—") : name;
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
        setWindowTitle(QStringLiteral("SeedVR2 Runner"));
        setMinimumSize(1280, 820);
        qApp->setStyleSheet(QString::fromLatin1(kStyle));
        buildUi();
        loadSettings();

        connect(m_engine, &ProcessEngine::logLine, this, &MainWindow::appendLog);
        connect(m_engine, &ProcessEngine::fileProgressUpdated, this, &MainWindow::onFileProgress);
        connect(m_engine, &ProcessEngine::processingFinished, this, &MainWindow::onFinished);
        connect(m_engine, &ProcessEngine::batchProgressUpdated, this, [this](int c, int t) {
            if (t > 0 && m_fileBar->value() == 0) m_fileBar->setValue((c * 100) / t);
        });

        setRunning(false);
    }

    ~MainWindow() override = default;

protected:
    void closeEvent(QCloseEvent *event) override
    {
        saveSettings();
        if (m_engine->isRunning()) m_engine->stopProcess();
        QMainWindow::closeEvent(event);
    }

private slots:
    void browseInput()
    {
        QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Input Directory"), m_inputPath->text());
        if (path.isEmpty()) {
            path = QFileDialog::getOpenFileName(this, QStringLiteral("Select Input File"), m_inputPath->text(),
                                                QStringLiteral("Media (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.mov *.mkv *.avi);;All files (*)"));
        }
        if (!path.isEmpty()) {
            m_inputPath->setText(path);
            refreshTimeline();
        }
    }

    void browseOutput()
    {
        const QString path = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), m_outputPath->text());
        if (!path.isEmpty()) m_outputPath->setText(path);
    }

    void browsePython()
    {
        const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Select Python Executable"), m_pythonExe->text(), QStringLiteral("Executable (*)"));
        if (!path.isEmpty()) m_pythonExe->setText(path);
    }

    void browseScript()
    {
        const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Select inference_cli.py"), m_scriptPath->text(), QStringLiteral("Python (*.py)"));
        if (!path.isEmpty()) m_scriptPath->setText(path);
    }

    void toggleSettings(bool expanded)
    {
        m_settingsContent->setVisible(expanded);
        m_toggleSettingsBtn->setText(expanded ? QStringLiteral("▼ Settings") : QStringLiteral("▶ Settings"));
    }

    void startRun()
    {
        const QString input = m_inputPath->text().trimmed();
        const QString output = m_outputPath->text().trimmed();
        const QString python = m_pythonExe->text().trimmed();
        const QString script = m_scriptPath->text().trimmed();
        if (input.isEmpty() || output.isEmpty() || python.isEmpty() || script.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("Missing Required Fields"), QStringLiteral("Input, output, Python executable, and CLI script are required."));
            return;
        }

        saveSettings();
        setRunning(true);
        m_log->clear();
        m_fileBar->setValue(0);
        m_queueBar->setValue(0);
        m_fileInfo->setText(QStringLiteral("Current: — (0/0)"));
        m_queueInfo->setText(QStringLiteral("Queue: 0/0"));

        QStringList args;
        args << QStringLiteral("--input") << input
             << QStringLiteral("--output") << output
             << QStringLiteral("--output-resolution") << m_outputResolution->currentText().trimmed()
             << QStringLiteral("--resize-method") << m_resizeMethod->currentText()
             << QStringLiteral("--ai-model") << m_aiModel->currentText()
             << QStringLiteral("--recover-detail") << QString::number(m_recoverDetail->value())
             << QStringLiteral("--grain") << QString::number(m_grain->value())
             << QStringLiteral("--batch-flush-interval") << QString::number(1);

        m_engine->startProcess(python, script, args);
        statusBar()->showMessage(QStringLiteral("Running..."));
    }

    void stopRun()
    {
        m_engine->stopProcess();
        statusBar()->showMessage(QStringLiteral("Stopping..."));
    }

    void appendLog(const QString &line)
    {
        m_log->appendPlainText(line);
        auto *sb = m_log->verticalScrollBar();
        sb->setValue(sb->maximum());
    }

    void onFileProgress(const QString &filePath, int current, int total, int doneFiles, int remainingFiles, int)
    {
        if (total > 0) m_fileBar->setValue((current * 100) / total);
        const int totalFiles = doneFiles + remainingFiles + 1;
        if (totalFiles > 0) m_queueBar->setValue((doneFiles * 100) / totalFiles);

        m_fileInfo->setText(QStringLiteral("Current: %1 (%2/%3)").arg(humanName(filePath)).arg(current).arg(total));
        m_queueInfo->setText(QStringLiteral("Queue: %1/%2").arg(doneFiles).arg(totalFiles));
    }

    void onFinished(bool success, const QString &message)
    {
        setRunning(false);
        if (success) {
            m_fileBar->setValue(100);
            m_queueBar->setValue(100);
        }
        statusBar()->showMessage(message, 6000);
    }

    void refreshTimeline()
    {
        qDeleteAll(m_timelineThumbs);
        m_timelineThumbs.clear();

        const QStringList files = collectImageFiles(m_inputPath->text().trimmed());
        const int limit = qMin(files.size(), 64);
        for (int i = 0; i < limit; ++i) {
            QPixmap pix(files.at(i));
            if (pix.isNull()) continue;

            auto *thumb = new QLabel(m_timelineContent);
            thumb->setObjectName(QStringLiteral("thumbnailLabel"));
            thumb->setFixedSize(120, 72);
            thumb->setAlignment(Qt::AlignCenter);
            thumb->setPixmap(pix.scaled(114, 66, Qt::KeepAspectRatio, Qt::SmoothTransformation));
            thumb->setToolTip(QFileInfo(files.at(i)).fileName());
            m_timelineLayout->insertWidget(m_timelineLayout->count() - 1, thumb);
            m_timelineThumbs.push_back(thumb);

            if (i == 0) {
                m_previewLabel->setPixmap(pix.scaled(m_previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }
        }
        if (m_timelineThumbs.isEmpty()) {
            m_previewLabel->setText(QStringLiteral("No preview frames available"));
        }
    }

private:
    void buildUi()
    {
        auto *central = new QWidget(this);
        setCentralWidget(central);

        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(12, 12, 12, 12);
        root->setSpacing(10);

        auto *pathRow = new QHBoxLayout();
        m_inputPath = new QLineEdit(this);
        m_inputPath->setPlaceholderText(QStringLiteral("Input video, image, or frame directory"));
        auto *inputBtn = new QPushButton(QStringLiteral("Input..."), this);
        m_outputPath = new QLineEdit(this);
        m_outputPath->setPlaceholderText(QStringLiteral("Output directory"));
        auto *outputBtn = new QPushButton(QStringLiteral("Output..."), this);
        m_pythonExe = new QLineEdit(this);
        m_pythonExe->setPlaceholderText(QStringLiteral("Python executable"));
        auto *pythonBtn = new QPushButton(QStringLiteral("Python..."), this);
        m_scriptPath = new QLineEdit(this);
        m_scriptPath->setPlaceholderText(QStringLiteral("inference_cli.py path"));
        auto *scriptBtn = new QPushButton(QStringLiteral("Script..."), this);

        pathRow->addWidget(m_inputPath, 2);
        pathRow->addWidget(inputBtn);
        pathRow->addWidget(m_outputPath, 2);
        pathRow->addWidget(outputBtn);
        pathRow->addWidget(m_pythonExe, 2);
        pathRow->addWidget(pythonBtn);
        pathRow->addWidget(m_scriptPath, 2);
        pathRow->addWidget(scriptBtn);
        root->addLayout(pathRow);

        connect(inputBtn, &QPushButton::clicked, this, &MainWindow::browseInput);
        connect(outputBtn, &QPushButton::clicked, this, &MainWindow::browseOutput);
        connect(pythonBtn, &QPushButton::clicked, this, &MainWindow::browsePython);
        connect(scriptBtn, &QPushButton::clicked, this, &MainWindow::browseScript);
        connect(m_inputPath, &QLineEdit::editingFinished, this, &MainWindow::refreshTimeline);

        auto *mainPanel = new QHBoxLayout();

        auto *previewFrame = new QFrame(this);
        previewFrame->setObjectName(QStringLiteral("previewFrame"));
        auto *previewLayout = new QVBoxLayout(previewFrame);
        auto *previewTitle = new QLabel(QStringLiteral("Preview"), previewFrame);
        previewTitle->setStyleSheet(QStringLiteral("font-size:15px;font-weight:600;"));
        m_previewLabel = new QLabel(QStringLiteral("No preview frames available"), previewFrame);
        m_previewLabel->setAlignment(Qt::AlignCenter);
        m_previewLabel->setMinimumSize(720, 420);
        m_previewLabel->setStyleSheet(QStringLiteral("background:#0f1116;border:1px solid #2f3441;border-radius:8px;"));
        previewLayout->addWidget(previewTitle);
        previewLayout->addWidget(m_previewLabel, 1);

        auto *settingsFrame = new QFrame(this);
        settingsFrame->setObjectName(QStringLiteral("settingsFrame"));
        settingsFrame->setMinimumWidth(340);
        auto *settingsLayout = new QVBoxLayout(settingsFrame);
        m_toggleSettingsBtn = new QToolButton(settingsFrame);
        m_toggleSettingsBtn->setCheckable(true);
        m_toggleSettingsBtn->setChecked(true);
        m_toggleSettingsBtn->setToolButtonStyle(Qt::ToolButtonTextOnly);
        settingsLayout->addWidget(m_toggleSettingsBtn);

        m_settingsContent = new QWidget(settingsFrame);
        auto *form = new QFormLayout(m_settingsContent);

        m_outputResolution = new QComboBox(m_settingsContent);
        m_outputResolution->setEditable(true);
        m_outputResolution->addItems({QStringLiteral("1280x720"), QStringLiteral("1920x1080"), QStringLiteral("2560x1440"), QStringLiteral("3840x2160")});
        m_outputResolution->setCurrentText(QStringLiteral("1920x1080"));

        m_resizeMethod = new QComboBox(m_settingsContent);
        m_resizeMethod->addItems({QStringLiteral("lanczos"), QStringLiteral("bicubic"), QStringLiteral("bilinear"), QStringLiteral("nearest")});

        m_aiModel = new QComboBox(m_settingsContent);
        m_aiModel->setEditable(true);
        m_aiModel->addItems({QStringLiteral("seedvr2-pro"), QStringLiteral("seedvr2-balanced"), QStringLiteral("seedvr2-fast")});

        m_recoverDetail = new QSlider(Qt::Horizontal, m_settingsContent);
        m_recoverDetail->setRange(0, 100);
        m_recoverDetail->setValue(35);
        m_recoverLabel = new QLabel(QString::number(m_recoverDetail->value()), m_settingsContent);

        m_grain = new QSlider(Qt::Horizontal, m_settingsContent);
        m_grain->setRange(0, 100);
        m_grain->setValue(8);
        m_grainLabel = new QLabel(QString::number(m_grain->value()), m_settingsContent);

        auto *detailRow = new QWidget(m_settingsContent);
        auto *detailLayout = new QHBoxLayout(detailRow);
        detailLayout->setContentsMargins(0, 0, 0, 0);
        detailLayout->addWidget(m_recoverDetail);
        detailLayout->addWidget(m_recoverLabel);

        auto *grainRow = new QWidget(m_settingsContent);
        auto *grainLayout = new QHBoxLayout(grainRow);
        grainLayout->setContentsMargins(0, 0, 0, 0);
        grainLayout->addWidget(m_grain);
        grainLayout->addWidget(m_grainLabel);

        form->addRow(QStringLiteral("Output Resolution"), m_outputResolution);
        form->addRow(QStringLiteral("Resize Method"), m_resizeMethod);
        form->addRow(QStringLiteral("AI Model"), m_aiModel);
        form->addRow(QStringLiteral("Recover Detail"), detailRow);
        form->addRow(QStringLiteral("Grain"), grainRow);

        settingsLayout->addWidget(m_settingsContent);
        settingsLayout->addStretch();

        connect(m_toggleSettingsBtn, &QToolButton::toggled, this, &MainWindow::toggleSettings);
        connect(m_recoverDetail, &QSlider::valueChanged, this, [this](int v) { m_recoverLabel->setText(QString::number(v)); });
        connect(m_grain, &QSlider::valueChanged, this, [this](int v) { m_grainLabel->setText(QString::number(v)); });
        toggleSettings(true);

        mainPanel->addWidget(previewFrame, 3);
        mainPanel->addWidget(settingsFrame, 1);
        root->addLayout(mainPanel, 1);

        auto *timelineFrame = new QFrame(this);
        timelineFrame->setObjectName(QStringLiteral("timelineFrame"));
        auto *timelineLayout = new QVBoxLayout(timelineFrame);
        auto *timelineTitle = new QLabel(QStringLiteral("Timeline"), timelineFrame);
        timelineTitle->setStyleSheet(QStringLiteral("font-size:14px;font-weight:600;"));

        m_timelineScroll = new QScrollArea(timelineFrame);
        m_timelineScroll->setWidgetResizable(true);
        m_timelineScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        m_timelineScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_timelineScroll->setMinimumHeight(122);

        m_timelineContent = new QWidget(m_timelineScroll);
        m_timelineLayout = new QHBoxLayout(m_timelineContent);
        m_timelineLayout->setContentsMargins(8, 8, 8, 8);
        m_timelineLayout->setSpacing(8);
        m_timelineLayout->addStretch();
        m_timelineScroll->setWidget(m_timelineContent);

        m_fileInfo = new QLabel(QStringLiteral("Current: — (0/0)"), timelineFrame);
        m_fileBar = new QProgressBar(timelineFrame);
        m_fileBar->setRange(0, 100);
        m_fileBar->setValue(0);

        m_queueInfo = new QLabel(QStringLiteral("Queue: 0/0"), timelineFrame);
        m_queueBar = new QProgressBar(timelineFrame);
        m_queueBar->setRange(0, 100);
        m_queueBar->setValue(0);

        timelineLayout->addWidget(timelineTitle);
        timelineLayout->addWidget(m_timelineScroll);
        timelineLayout->addWidget(m_fileInfo);
        timelineLayout->addWidget(m_fileBar);
        timelineLayout->addWidget(m_queueInfo);
        timelineLayout->addWidget(m_queueBar);
        root->addWidget(timelineFrame);

        auto *logFrame = new QFrame(this);
        logFrame->setObjectName(QStringLiteral("logFrame"));
        auto *logLayout = new QVBoxLayout(logFrame);
        auto *logTitle = new QLabel(QStringLiteral("Log"), logFrame);
        logTitle->setStyleSheet(QStringLiteral("font-size:14px;font-weight:600;"));
        m_log = new QPlainTextEdit(logFrame);
        m_log->setReadOnly(true);
        m_log->setMaximumBlockCount(5000);
        m_log->setMinimumHeight(170);
        logLayout->addWidget(logTitle);
        logLayout->addWidget(m_log);
        root->addWidget(logFrame);

        auto *actions = new QHBoxLayout();
        actions->addStretch();
        m_startBtn = new QPushButton(QStringLiteral("Start"), this);
        m_startBtn->setObjectName(QStringLiteral("startBtn"));
        m_stopBtn = new QPushButton(QStringLiteral("Stop"), this);
        m_stopBtn->setObjectName(QStringLiteral("stopBtn"));
        actions->addWidget(m_startBtn);
        actions->addWidget(m_stopBtn);
        root->addLayout(actions);

        connect(m_startBtn, &QPushButton::clicked, this, &MainWindow::startRun);
        connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::stopRun);

        statusBar()->showMessage(QStringLiteral("Ready"));
    }

    void setRunning(bool running)
    {
        m_startBtn->setEnabled(!running);
        m_stopBtn->setEnabled(running);
        m_inputPath->setEnabled(!running);
        m_outputPath->setEnabled(!running);
        m_pythonExe->setEnabled(!running);
        m_scriptPath->setEnabled(!running);
        m_toggleSettingsBtn->setEnabled(!running);
        m_settingsContent->setEnabled(!running);
    }

    void saveSettings()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("Runner"));
        s.setValue(QStringLiteral("inputPath"), m_inputPath->text());
        s.setValue(QStringLiteral("outputPath"), m_outputPath->text());
        s.setValue(QStringLiteral("pythonExe"), m_pythonExe->text());
        s.setValue(QStringLiteral("scriptPath"), m_scriptPath->text());
        s.setValue(QStringLiteral("outputResolution"), m_outputResolution->currentText());
        s.setValue(QStringLiteral("resizeMethod"), m_resizeMethod->currentText());
        s.setValue(QStringLiteral("aiModel"), m_aiModel->currentText());
        s.setValue(QStringLiteral("recoverDetail"), m_recoverDetail->value());
        s.setValue(QStringLiteral("grain"), m_grain->value());
    }

    void loadSettings()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("Runner"));
        m_inputPath->setText(s.value(QStringLiteral("inputPath")).toString());
        m_outputPath->setText(s.value(QStringLiteral("outputPath")).toString());
        m_pythonExe->setText(s.value(QStringLiteral("pythonExe")).toString());
        m_scriptPath->setText(s.value(QStringLiteral("scriptPath")).toString());

        const QString outputResolution = s.value(QStringLiteral("outputResolution"), QStringLiteral("1920x1080")).toString();
        m_outputResolution->setCurrentText(outputResolution);

        const QString resizeMethod = s.value(QStringLiteral("resizeMethod"), QStringLiteral("lanczos")).toString();
        {
            QSignalBlocker blocker(m_resizeMethod);
            int idx = m_resizeMethod->findText(resizeMethod);
            if (idx < 0) {
                m_resizeMethod->addItem(resizeMethod);
                idx = m_resizeMethod->findText(resizeMethod);
            }
            m_resizeMethod->setCurrentIndex(idx);
        }

        const QString aiModel = s.value(QStringLiteral("aiModel"), QStringLiteral("seedvr2-pro")).toString();
        m_aiModel->setCurrentText(aiModel);
        m_recoverDetail->setValue(s.value(QStringLiteral("recoverDetail"), 35).toInt());
        m_grain->setValue(s.value(QStringLiteral("grain"), 8).toInt());
        refreshTimeline();
    }

    QLineEdit *m_inputPath = nullptr;
    QLineEdit *m_outputPath = nullptr;
    QLineEdit *m_pythonExe = nullptr;
    QLineEdit *m_scriptPath = nullptr;

    QLabel *m_previewLabel = nullptr;

    QToolButton *m_toggleSettingsBtn = nullptr;
    QWidget *m_settingsContent = nullptr;
    QComboBox *m_outputResolution = nullptr;
    QComboBox *m_resizeMethod = nullptr;
    QComboBox *m_aiModel = nullptr;
    QSlider *m_recoverDetail = nullptr;
    QSlider *m_grain = nullptr;
    QLabel *m_recoverLabel = nullptr;
    QLabel *m_grainLabel = nullptr;

    QScrollArea *m_timelineScroll = nullptr;
    QWidget *m_timelineContent = nullptr;
    QHBoxLayout *m_timelineLayout = nullptr;
    QList<QLabel *> m_timelineThumbs;

    QLabel *m_fileInfo = nullptr;
    QLabel *m_queueInfo = nullptr;
    QProgressBar *m_fileBar = nullptr;
    QProgressBar *m_queueBar = nullptr;

    QPlainTextEdit *m_log = nullptr;

    QPushButton *m_startBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;

    ProcessEngine *m_engine = nullptr;
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("SeedVR2 Runner"));
    app.setOrganizationName(QStringLiteral("SeedVR2"));

    MainWindow w;
    w.show();
    return app.exec();
}
