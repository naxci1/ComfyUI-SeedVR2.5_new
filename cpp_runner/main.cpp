#include "ProcessEngine.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QToolBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QUrl>
#include <QtGlobal>

namespace {

constexpr const char *kStyleSheet = R"(
QWidget { background-color:#151923; color:#e8eeff; font-family:"Inter","Segoe UI",sans-serif; font-size:13px; }
QMainWindow { background-color:#0e1118; }
QFrame#card, QFrame#previewPanel, QFrame#settingsPanel, QFrame#progressPanel, QFrame#timelinePanel, QFrame#playbackPanel { background-color:#1d2331; border:1px solid #2f3850; border-radius:12px; }
QLabel#panelTitle { font-size:15px; font-weight:700; color:#f3f7ff; }
QLineEdit, QComboBox, QSpinBox { background-color:#111726; border:1px solid #3c4868; border-radius:8px; padding:6px 8px; }
QPushButton { background-color:#2e3b5e; border:1px solid #45598a; border-radius:8px; padding:6px 12px; color:#edf3ff; font-weight:600; }
QPushButton:hover { background-color:#3b4f7a; }
QPushButton#startButton { background-color:#146dff; border:none; color:white; font-size:14px; }
QPushButton#stopButton { background-color:#8b2c40; border:none; color:white; font-size:14px; }
QGroupBox { border:1px solid #2f3850; border-radius:10px; margin-top:10px; padding-top:8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
QToolBox::tab { background:#111726; border:1px solid #2f3850; border-radius:8px; padding:8px; font-weight:700; }
QToolBox::tab:selected { background:#243151; }
QProgressBar { background:#0f1523; border:1px solid #32446c; border-radius:8px; text-align:center; min-height:20px; }
QProgressBar::chunk { border-radius:8px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #03ffd5, stop:0.5 #2f8cff, stop:1 #7a5cff); }
QPlainTextEdit { background:#0a0f18; border:1px solid #2d3958; border-radius:10px; color:#dbe8ff; font-family:"JetBrains Mono","Consolas",monospace; font-size:12px; }
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
    if (!target) return;

    if (path.trimmed().isEmpty() || !QFileInfo::exists(path)) {
        target->setText(QStringLiteral("Drop media or choose input"));
        target->setPixmap(QPixmap());
        return;
    }

    const QPixmap px(path);
    if (px.isNull()) {
        target->setText(QStringLiteral("Preview unavailable for this file type"));
        target->setPixmap(QPixmap());
        return;
    }

    target->setPixmap(px.scaled(target->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    target->setText(QString());
}

QString extractGpuId(const QString &entry)
{
    if (entry == QStringLiteral("Auto")) return QStringLiteral("0");
    if (entry == QStringLiteral("CPU")) return QStringLiteral("cpu");

    if (entry.startsWith(QStringLiteral("GPU "))) {
        QString token = entry;
        const int colonPos = token.indexOf(QLatin1Char(':'));
        if (colonPos >= 0) token = token.left(colonPos);
        const QStringList parts = token.split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (!parts.isEmpty()) return parts.last().trimmed();
    }

    return QStringLiteral("0");
}

QString crfForQuality(const QString &quality)
{
    if (quality == QStringLiteral("Low")) return QStringLiteral("28");
    if (quality == QStringLiteral("High")) return QStringLiteral("18");
    return QStringLiteral("23");
}

} // namespace

class SettingsDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr)
        : QDialog(parent)
    {
        setWindowTitle(QStringLiteral("Paths & Configuration"));
        resize(760, 420);

        auto *root = new QVBoxLayout(this);

        auto *pathsGroup = new QGroupBox(QStringLiteral("Paths"), this);
        auto *form = new QFormLayout(pathsGroup);

        m_pythonExeEdit = new QLineEdit(pathsGroup);
        m_seedvrFolderEdit = new QLineEdit(pathsGroup);
        m_inputEdit = new QLineEdit(pathsGroup);
        m_outputEdit = new QLineEdit(pathsGroup);
        m_modelDirEdit = new QLineEdit(pathsGroup);

        auto *pyBtn = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        auto *seedBtn = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        auto *inBtn = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        auto *outBtn = new QPushButton(QStringLiteral("Browse"), pathsGroup);
        auto *modelBtn = new QPushButton(QStringLiteral("Browse"), pathsGroup);

        form->addRow(QStringLiteral("Python Executable:"), buildPathRow(pathsGroup, m_pythonExeEdit, pyBtn));
        form->addRow(QStringLiteral("SeedVR2 Folder:"), buildPathRow(pathsGroup, m_seedvrFolderEdit, seedBtn));
        form->addRow(QStringLiteral("Input Path:"), buildPathRow(pathsGroup, m_inputEdit, inBtn));
        form->addRow(QStringLiteral("Output Path:"), buildPathRow(pathsGroup, m_outputEdit, outBtn));
        form->addRow(QStringLiteral("Model Directory:"), buildPathRow(pathsGroup, m_modelDirEdit, modelBtn));

        root->addWidget(pathsGroup);

        auto *gpuGroup = new QGroupBox(QStringLiteral("GPU Management"), this);
        auto *gpuForm = new QFormLayout(gpuGroup);
        m_primaryGpu = new QComboBox(gpuGroup);
        m_secondaryGpu = new QComboBox(gpuGroup);
        auto *refreshGpuBtn = new QPushButton(QStringLiteral("Refresh CUDA Devices"), gpuGroup);

        gpuForm->addRow(QStringLiteral("Primary GPU:"), m_primaryGpu);
        gpuForm->addRow(QStringLiteral("Secondary GPU:"), m_secondaryGpu);
        gpuForm->addRow(QString(), refreshGpuBtn);

        root->addWidget(gpuGroup);

        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
        root->addWidget(buttons);

        connect(pyBtn, &QPushButton::clicked, this, [this]() {
            const QString f = QFileDialog::getOpenFileName(this, QStringLiteral("Select Python Executable"), m_pythonExeEdit->text());
            if (!f.isEmpty()) m_pythonExeEdit->setText(f);
        });
        connect(seedBtn, &QPushButton::clicked, this, [this]() {
            const QString d = QFileDialog::getExistingDirectory(this, QStringLiteral("Select SeedVR2 Folder"), m_seedvrFolderEdit->text());
            if (!d.isEmpty()) m_seedvrFolderEdit->setText(d);
        });
        connect(inBtn, &QPushButton::clicked, this, [this]() {
            const QString f = QFileDialog::getOpenFileName(this, QStringLiteral("Select Input"), m_inputEdit->text(), QStringLiteral("Media (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
            if (!f.isEmpty()) m_inputEdit->setText(f);
        });
        connect(outBtn, &QPushButton::clicked, this, [this]() {
            const QString d = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Directory"), m_outputEdit->text());
            if (!d.isEmpty()) m_outputEdit->setText(d);
        });
        connect(modelBtn, &QPushButton::clicked, this, [this]() {
            const QString d = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Model Directory"), m_modelDirEdit->text());
            if (!d.isEmpty()) m_modelDirEdit->setText(d);
        });

        connect(refreshGpuBtn, &QPushButton::clicked, this, &SettingsDialog::refreshGpuList);
        connect(buttons, &QDialogButtonBox::accepted, this, &SettingsDialog::onSaveClicked);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

        load();
        refreshGpuList();
    }

    QString pythonExe() const { return m_pythonExeEdit->text().trimmed(); }
    QString seedvrFolder() const { return m_seedvrFolderEdit->text().trimmed(); }
    QString inputPath() const { return m_inputEdit->text().trimmed(); }
    QString outputPath() const { return m_outputEdit->text().trimmed(); }
    QString modelDir() const { return m_modelDirEdit->text().trimmed(); }
    QString primaryGpu() const { return m_primaryGpu->currentText(); }
    QString secondaryGpu() const { return m_secondaryGpu->currentText(); }

signals:
    void settingsChanged();

private slots:
    void refreshGpuList()
    {
        const QString oldPrimary = m_primaryGpu->currentText();
        const QString oldSecondary = m_secondaryGpu->currentText();

        m_primaryGpu->clear();
        m_secondaryGpu->clear();

        QStringList devices;
        devices << QStringLiteral("Auto") << QStringLiteral("CPU");

        QString python = pythonExe();
        if (python.isEmpty()) python = QStringLiteral("python");

        QProcess probe;
        QStringList args;
        args << QStringLiteral("-c")
             << QStringLiteral("import torch;"
                               "print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NONE');"
                               "n=torch.cuda.device_count() if torch.cuda.is_available() else 0;"
                               "[print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(n)]");
        probe.start(python, args);
        probe.waitForFinished(2500);

        const QString out = QString::fromUtf8(probe.readAllStandardOutput());
        const QStringList lines = out.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        for (const QString &line : lines) {
            if (line.startsWith(QStringLiteral("GPU "))) {
                devices << line.trimmed();
            }
        }

        if (devices.size() == 2) {
            devices << QStringLiteral("GPU 0");
        }

        m_primaryGpu->addItems(devices);
        m_secondaryGpu->addItems(devices);
        m_secondaryGpu->setCurrentText(QStringLiteral("Auto"));

        if (!oldPrimary.isEmpty()) m_primaryGpu->setCurrentText(oldPrimary);
        if (!oldSecondary.isEmpty()) m_secondaryGpu->setCurrentText(oldSecondary);
    }

    void onSaveClicked()
    {
        save();
        emit settingsChanged();
        accept();
    }

private:
    void load()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        m_pythonExeEdit->setText(s.value(QStringLiteral("settings/pythonExe"), QStringLiteral("python")).toString());
        m_seedvrFolderEdit->setText(s.value(QStringLiteral("settings/seedvrFolder")).toString());
        m_inputEdit->setText(s.value(QStringLiteral("settings/inputPath")).toString());
        m_outputEdit->setText(s.value(QStringLiteral("settings/outputPath")).toString());
        m_modelDirEdit->setText(s.value(QStringLiteral("settings/modelDir")).toString());
    }

    void save()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        s.setValue(QStringLiteral("settings/pythonExe"), m_pythonExeEdit->text());
        s.setValue(QStringLiteral("settings/seedvrFolder"), m_seedvrFolderEdit->text());
        s.setValue(QStringLiteral("settings/inputPath"), m_inputEdit->text());
        s.setValue(QStringLiteral("settings/outputPath"), m_outputEdit->text());
        s.setValue(QStringLiteral("settings/modelDir"), m_modelDirEdit->text());
        s.setValue(QStringLiteral("settings/primaryGpu"), m_primaryGpu->currentText());
        s.setValue(QStringLiteral("settings/secondaryGpu"), m_secondaryGpu->currentText());
    }

    QLineEdit *m_pythonExeEdit = nullptr;
    QLineEdit *m_seedvrFolderEdit = nullptr;
    QLineEdit *m_inputEdit = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QLineEdit *m_modelDirEdit = nullptr;
    QComboBox *m_primaryGpu = nullptr;
    QComboBox *m_secondaryGpu = nullptr;
};

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr)
        : QMainWindow(parent)
        , m_engine(new ProcessEngine(this))
        , m_settingsDialog(new SettingsDialog(this))
    {
        setWindowTitle(QStringLiteral("SeedVR2 Pro Upscaler"));
        setMinimumSize(1650, 980);
        setAcceptDrops(true);

        if (qApp) qApp->setStyleSheet(QString::fromLatin1(kStyleSheet));

        buildUi();
        loadUiSettings();

        connect(m_engine, &ProcessEngine::logLine, this, &MainWindow::appendLogLine);
        connect(m_engine, &ProcessEngine::fileProgressUpdated, this, &MainWindow::onFileProgress);
        connect(m_engine, &ProcessEngine::batchProgressUpdated, this, &MainWindow::onBatchProgress);
        connect(m_engine, &ProcessEngine::processingFinished, this, &MainWindow::onProcessingFinished);
        connect(m_settingsDialog, &SettingsDialog::settingsChanged, this, &MainWindow::applySettingsDialogValues);

        applySettingsDialogValues();
        setRunning(false);
    }

    ~MainWindow() override = default;

protected:
    void closeEvent(QCloseEvent *event) override
    {
        saveUiSettings();
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
        const QString path = urls.first().toLocalFile();
        if (path.isEmpty()) return;

        m_inputPreviewPath->setText(path);
        if (m_settingsDialog->inputPath().isEmpty()) {
            QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
            s.setValue(QStringLiteral("settings/inputPath"), path);
        }
        refreshPreviewPanels();
        appendLogLine(QStringLiteral("Dropped input: %1").arg(path));
    }

private slots:
    void openSettingsDialog()
    {
        m_settingsDialog->show();
        m_settingsDialog->raise();
        m_settingsDialog->activateWindow();
    }

    void browsePreviewInput()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Input Preview"),
            m_inputPreviewPath->text(),
            QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
        if (!path.isEmpty()) {
            m_inputPreviewPath->setText(path);
            refreshPreviewPanels();
        }
    }

    void browsePreviewOutput()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select Output Preview"),
            m_outputPreviewPath->text(),
            QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"));
        if (!path.isEmpty()) {
            m_outputPreviewPath->setText(path);
            refreshPreviewPanels();
        }
    }

    void browseLutPath()
    {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Select LUT"),
            m_lutPathEdit->text(),
            QStringLiteral("LUT Files (*.cube *.3dl *.png *.jpg *.jpeg *.bmp);;All files (*)"));
        if (!path.isEmpty()) m_lutPathEdit->setText(path);
    }

    void refreshPreviewPanels()
    {
        setPreviewPixmap(m_splitInputPreview, m_inputPreviewPath->text().trimmed());
        setPreviewPixmap(m_inputPreviewSolo, m_inputPreviewPath->text().trimmed());
        setPreviewPixmap(m_splitOutputPreview, m_outputPreviewPath->text().trimmed());
        setPreviewPixmap(m_outputPreviewSolo, m_outputPreviewPath->text().trimmed());
    }

    void updateResolutionMode(const QString &mode)
    {
        const bool pixel = (mode == QStringLiteral("Pixel"));
        m_resolutionSpin->setVisible(pixel);
        m_resolutionScaleCombo->setVisible(!pixel);
    }

    void startRender()
    {
        const QString scriptPath = findDefaultCliScriptPath();
        if (!QFileInfo::exists(scriptPath)) {
            QMessageBox::critical(this, QStringLiteral("CLI Not Found"), QStringLiteral("Could not locate inference_cli.py"));
            return;
        }

        const QString input = m_settingsDialog->inputPath();
        if (input.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("Missing Input"), QStringLiteral("Set input path in Edit > Settings."));
            return;
        }

        QString output = m_settingsDialog->outputPath();
        if (output.isEmpty()) {
            output = QDir::current().absoluteFilePath(QStringLiteral("output"));
        }

        QString pythonExe = m_settingsDialog->pythonExe();
        if (pythonExe.isEmpty()) pythonExe = QStringLiteral("python");

        m_logConsole->clear();
        m_fileProgress->setValue(0);
        m_batchProgress->setValue(0);
        m_fileLabel->setText(QStringLiteral("Current File: — (0/0)"));
        m_batchLabel->setText(QStringLiteral("Batch Progress: 0/0"));

        QStringList args;
        args << input;
        args << QStringLiteral("--output") << output;

        const QString modelDir = m_settingsDialog->modelDir();
        if (!modelDir.isEmpty()) args << QStringLiteral("--model_dir") << modelDir;

        args << QStringLiteral("--dit_model") << m_ditModelCombo->currentText();

        const int preDownscale = m_preDownscaleCombo->currentText().left(1).toInt();
        if (preDownscale > 1) args << QStringLiteral("--pre_downscale") << QString::number(preDownscale);

        if (m_resolutionModeCombo->currentText() == QStringLiteral("X Times")) {
            args << QStringLiteral("--resolution_mode") << QStringLiteral("xtimes");
            const int times = m_resolutionScaleCombo->currentText().left(1).toInt();
            args << QStringLiteral("--resolution_scale") << QString::number(times);
        } else {
            args << QStringLiteral("--resolution") << QString::number(m_resolutionSpin->value());
        }

        if (m_maxResolutionSpin->value() > 0) args << QStringLiteral("--max_resolution") << QString::number(m_maxResolutionSpin->value());
        args << QStringLiteral("--batch_size") << QString::number(m_batchSizeSpin->value());
        if (m_uniformBatchCheck->isChecked()) args << QStringLiteral("--uniform_batch_size");
        args << QStringLiteral("--seed") << QString::number(m_seedSpin->value());
        if (m_skipFirstFramesSpin->value() > 0) args << QStringLiteral("--skip_first_frames") << QString::number(m_skipFirstFramesSpin->value());
        if (m_loadCapSpin->value() > 0) args << QStringLiteral("--load_cap") << QString::number(m_loadCapSpin->value());
        if (m_chunkSizeSpin->value() > 0) args << QStringLiteral("--chunk_size") << QString::number(m_chunkSizeSpin->value());
        if (m_prependFramesSpin->value() > 0) args << QStringLiteral("--prepend_frames") << QString::number(m_prependFramesSpin->value());
        if (m_temporalOverlapSpin->value() > 0) args << QStringLiteral("--temporal_overlap") << QString::number(m_temporalOverlapSpin->value());

        if (m_colorCorrectionCombo->currentText() != QStringLiteral("lab")) {
            args << QStringLiteral("--color_correction") << m_colorCorrectionCombo->currentText();
        }
        if (m_attentionModeCombo->currentText() != QStringLiteral("sdpa")) {
            args << QStringLiteral("--attention_mode") << m_attentionModeCombo->currentText();
        }

        // Multi-GPU mapping from settings dialog (primary + secondary)
        const QString p = extractGpuId(m_settingsDialog->primaryGpu());
        const QString s = extractGpuId(m_settingsDialog->secondaryGpu());
        QString cudaDevice = p;
        if (p != QStringLiteral("cpu") && s != QStringLiteral("0") && s != QStringLiteral("cpu") && s != p) {
            cudaDevice = p + QStringLiteral(",") + s;
        } else if (p == QStringLiteral("0") && s.startsWith(QLatin1String("GPU"))) {
            const QString sid = extractGpuId(s);
            if (sid != p) cudaDevice = p + QStringLiteral(",") + sid;
        }
        args << QStringLiteral("--cuda_device") << cudaDevice;

        if (m_ditOffloadCombo->currentText() != QStringLiteral("none")) args << QStringLiteral("--dit_offload_device") << m_ditOffloadCombo->currentText();
        if (m_vaeOffloadCombo->currentText() != QStringLiteral("none")) args << QStringLiteral("--vae_offload_device") << m_vaeOffloadCombo->currentText();
        if (m_tensorOffloadCombo->currentText() != QStringLiteral("none")) args << QStringLiteral("--tensor_offload_device") << m_tensorOffloadCombo->currentText();

        if (m_blocksToSwapSpin->value() > 0) args << QStringLiteral("--blocks_to_swap") << QString::number(m_blocksToSwapSpin->value());
        if (m_swapIoCheck->isChecked()) args << QStringLiteral("--swap_io_components");

        if (m_vaeEncodeTiledCheck->isChecked()) {
            args << QStringLiteral("--vae_encode_tiled");
            if (m_vaeEncodeTileSizeSpin->value() != 1024) args << QStringLiteral("--vae_encode_tile_size") << QString::number(m_vaeEncodeTileSizeSpin->value());
            if (m_vaeEncodeTileOverlapSpin->value() != 128) args << QStringLiteral("--vae_encode_tile_overlap") << QString::number(m_vaeEncodeTileOverlapSpin->value());
        }

        if (m_vaeDecodeTiledCheck->isChecked()) {
            args << QStringLiteral("--vae_decode_tiled");
            if (m_vaeDecodeTileSizeSpin->value() != 1024) args << QStringLiteral("--vae_decode_tile_size") << QString::number(m_vaeDecodeTileSizeSpin->value());
            if (m_vaeDecodeTileOverlapSpin->value() != 128) args << QStringLiteral("--vae_decode_tile_overlap") << QString::number(m_vaeDecodeTileOverlapSpin->value());
        }

        if (m_tileDebugCombo->currentText() != QStringLiteral("false")) args << QStringLiteral("--tile_debug") << m_tileDebugCombo->currentText();
        if (m_cacheDitCheck->isChecked()) args << QStringLiteral("--cache_dit");
        if (m_cacheVaeCheck->isChecked()) args << QStringLiteral("--cache_vae");
        if (m_autoSafeguardCheck->isChecked()) args << QStringLiteral("--auto_safeguard");
        if (m_debugCheck->isChecked()) args << QStringLiteral("--debug");

        // Codec settings mapping (exact CLI arguments)
        const bool imageSeq = (m_outputTypeCombo->currentText() == QStringLiteral("Image Sequence"));
        if (imageSeq) {
            args << QStringLiteral("--output_format") << QStringLiteral("png");
        } else {
            args << QStringLiteral("--output_format") << m_containerCombo->currentText().toLower();
            args << QStringLiteral("--video_backend") << QStringLiteral("ffmpeg");

            QStringList ff;
            if (m_codecCombo->currentText() == QStringLiteral("H265")) {
                ff << QStringLiteral("-c:v") << QStringLiteral("libx265");
                args << QStringLiteral("--10bit");
            } else {
                ff << QStringLiteral("-c:v") << QStringLiteral("libx264");
            }
            ff << QStringLiteral("-preset") << QStringLiteral("medium");
            ff << QStringLiteral("-crf") << crfForQuality(m_qualityCombo->currentText());
            if (m_bitrateModeCombo->currentText() == QStringLiteral("Constant")) {
                ff << QStringLiteral("-b:v") << QStringLiteral("12M");
            }
            args << QStringLiteral("--ffmpeg_video_args") << QString::fromUtf8(QJsonDocument::fromVariant(ff).toJson(QJsonDocument::Compact));
        }

        if (!m_lutPathEdit->text().trimmed().isEmpty()) {
            args << QStringLiteral("--lut") << m_lutPathEdit->text().trimmed();
        }

        saveUiSettings();
        setRunning(true);
        statusBar()->showMessage(QStringLiteral("Running..."));
        m_engine->startProcess(pythonExe, scriptPath, args);
    }

    void stopRender()
    {
        m_engine->stopProcess();
        statusBar()->showMessage(QStringLiteral("Stopping..."));
    }

    void savePreset()
    {
        const QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save Preset"), QString(), QStringLiteral("JSON (*.json)"));
        if (path.isEmpty()) return;

        QJsonObject o;
        o["dit_model"] = m_ditModelCombo->currentText();
        o["pre_downscale"] = m_preDownscaleCombo->currentText();
        o["resolution_mode"] = m_resolutionModeCombo->currentText();
        o["resolution"] = m_resolutionSpin->value();
        o["resolution_scale"] = m_resolutionScaleCombo->currentText();
        o["max_resolution"] = m_maxResolutionSpin->value();
        o["batch_size"] = m_batchSizeSpin->value();
        o["uniform_batch"] = m_uniformBatchCheck->isChecked();
        o["seed"] = m_seedSpin->value();
        o["color_correction"] = m_colorCorrectionCombo->currentText();
        o["attention_mode"] = m_attentionModeCombo->currentText();
        o["output_type"] = m_outputTypeCombo->currentText();
        o["codec"] = m_codecCombo->currentText();
        o["bitrate_mode"] = m_bitrateModeCombo->currentText();
        o["quality"] = m_qualityCombo->currentText();
        o["container"] = m_containerCombo->currentText();
        o["lut"] = m_lutPathEdit->text();

        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) return;
        f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    }

    void loadPreset()
    {
        const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Load Preset"), QString(), QStringLiteral("JSON (*.json)"));
        if (path.isEmpty()) return;

        QFile f(path);
        if (!f.open(QIODevice::ReadOnly)) return;
        const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
        if (!doc.isObject()) return;
        const QJsonObject o = doc.object();

        if (o.contains("dit_model")) m_ditModelCombo->setCurrentText(o.value("dit_model").toString());
        if (o.contains("pre_downscale")) m_preDownscaleCombo->setCurrentText(o.value("pre_downscale").toString());
        if (o.contains("resolution_mode")) m_resolutionModeCombo->setCurrentText(o.value("resolution_mode").toString());
        if (o.contains("resolution")) m_resolutionSpin->setValue(o.value("resolution").toInt(720));
        if (o.contains("resolution_scale")) m_resolutionScaleCombo->setCurrentText(o.value("resolution_scale").toString());
        if (o.contains("max_resolution")) m_maxResolutionSpin->setValue(o.value("max_resolution").toInt(0));
        if (o.contains("batch_size")) m_batchSizeSpin->setValue(o.value("batch_size").toInt(81));
        if (o.contains("uniform_batch")) m_uniformBatchCheck->setChecked(o.value("uniform_batch").toBool(false));
        if (o.contains("seed")) m_seedSpin->setValue(o.value("seed").toInt(313));
        if (o.contains("color_correction")) m_colorCorrectionCombo->setCurrentText(o.value("color_correction").toString());
        if (o.contains("attention_mode")) m_attentionModeCombo->setCurrentText(o.value("attention_mode").toString());
        if (o.contains("output_type")) m_outputTypeCombo->setCurrentText(o.value("output_type").toString());
        if (o.contains("codec")) m_codecCombo->setCurrentText(o.value("codec").toString());
        if (o.contains("bitrate_mode")) m_bitrateModeCombo->setCurrentText(o.value("bitrate_mode").toString());
        if (o.contains("quality")) m_qualityCombo->setCurrentText(o.value("quality").toString());
        if (o.contains("container")) m_containerCombo->setCurrentText(o.value("container").toString());
        if (o.contains("lut")) m_lutPathEdit->setText(o.value("lut").toString());
    }

    void appendLogLine(const QString &line)
    {
        m_logConsole->appendPlainText(line);
        QScrollBar *scroll = m_logConsole->verticalScrollBar();
        scroll->setValue(scroll->maximum());
    }

    void onFileProgress(const QString &filename, int current, int total, int doneFiles, int remainingFiles, int)
    {
        const int pct = (total > 0) ? qBound(0, (current * 100) / total, 100) : 0;
        m_fileProgress->setValue(pct);
        m_fileLabel->setText(QStringLiteral("Current File: %1 (%2/%3)").arg(QFileInfo(filename).fileName()).arg(current).arg(total));

        const int totalFiles = doneFiles + remainingFiles + 1;
        const int batchPct = (totalFiles > 0) ? qBound(0, (doneFiles * 100) / totalFiles, 100) : 0;
        m_batchProgress->setValue(batchPct);
        m_batchLabel->setText(QStringLiteral("Batch Progress: %1/%2").arg(doneFiles).arg(totalFiles));

        if (!m_outputPreviewPath->text().trimmed().isEmpty()) refreshPreviewPanels();
    }

    void onBatchProgress(int current, int total)
    {
        if (total > 0) {
            m_batchProgress->setValue(qBound(0, (current * 100) / total, 100));
            m_batchLabel->setText(QStringLiteral("Batch Progress: %1/%2").arg(current).arg(total));
        }
    }

    void onProcessingFinished(bool success, const QString &message)
    {
        setRunning(false);
        statusBar()->showMessage(success ? QStringLiteral("Process completed") : QStringLiteral("Process stopped"), 5000);
        appendLogLine(QStringLiteral("Process result: %1").arg(message));
    }

    void applySettingsDialogValues()
    {
        m_inputPreviewPath->setText(m_settingsDialog->inputPath());
        refreshPreviewPanels();
        rebuildTimelineMarkers();
    }

private:
    void setRunning(bool running)
    {
        m_startBtn->setEnabled(!running);
        m_stopBtn->setEnabled(running);
    }

    void buildMenus()
    {
        auto *fileMenu = menuBar()->addMenu(QStringLiteral("File"));
        auto *editMenu = menuBar()->addMenu(QStringLiteral("Edit"));
        auto *procMenu = menuBar()->addMenu(QStringLiteral("Process"));
        menuBar()->addMenu(QStringLiteral("Account"));
        menuBar()->addMenu(QStringLiteral("Plugins"));
        auto *helpMenu = menuBar()->addMenu(QStringLiteral("Help"));

        auto *settingsAct = editMenu->addAction(QStringLiteral("Settings"));
        auto *exitAct = fileMenu->addAction(QStringLiteral("Exit"));
        auto *runAct = procMenu->addAction(QStringLiteral("Run"));
        auto *stopAct = procMenu->addAction(QStringLiteral("Stop"));
        helpMenu->addAction(QStringLiteral("About"), this, [this]() {
            QMessageBox::information(this, QStringLiteral("About"), QStringLiteral("SeedVR2 Qt6 Pro Runner"));
        });

        connect(settingsAct, &QAction::triggered, this, &MainWindow::openSettingsDialog);
        connect(exitAct, &QAction::triggered, this, &QWidget::close);
        connect(runAct, &QAction::triggered, this, &MainWindow::startRender);
        connect(stopAct, &QAction::triggered, this, &MainWindow::stopRender);
    }

    QWidget *buildCustomTitleBar(QWidget *parent)
    {
        auto *bar = new QFrame(parent);
        bar->setObjectName(QStringLiteral("card"));
        auto *layout = new QHBoxLayout(bar);
        layout->setContentsMargins(10, 6, 10, 6);

        auto *title = new QLabel(QStringLiteral("SeedVR2 Video Upscaler"), bar);
        title->setObjectName(QStringLiteral("panelTitle"));

        auto mkBtn = [bar](const QString &color) {
            auto *b = new QPushButton(bar);
            b->setFixedSize(14, 14);
            b->setStyleSheet(QStringLiteral("QPushButton{background:%1;border:none;border-radius:7px;}" ).arg(color));
            return b;
        };

        auto *closeBtn = mkBtn(QStringLiteral("#ff5f57"));
        auto *minBtn = mkBtn(QStringLiteral("#ffbd2e"));
        auto *maxBtn = mkBtn(QStringLiteral("#28c940"));

        layout->addWidget(title);
        layout->addStretch(1);
        layout->addWidget(minBtn);
        layout->addWidget(maxBtn);
        layout->addWidget(closeBtn);

        connect(closeBtn, &QPushButton::clicked, this, &QWidget::close);
        connect(minBtn, &QPushButton::clicked, this, &QWidget::showMinimized);
        connect(maxBtn, &QPushButton::clicked, this, [this]() {
            isMaximized() ? showNormal() : showMaximized();
        });

        return bar;
    }

    void buildUi()
    {
        buildMenus();

        auto *central = new QWidget(this);
        setCentralWidget(central);

        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(10, 10, 10, 10);
        root->setSpacing(8);

        root->addWidget(buildCustomTitleBar(central));

        auto *mainSplit = new QSplitter(Qt::Horizontal, central);
        root->addWidget(mainSplit, 1);

        // Left content: preview + playback + timeline + progress + log
        auto *left = new QWidget(mainSplit);
        auto *leftLayout = new QVBoxLayout(left);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        leftLayout->setSpacing(8);

        auto *previewFrame = new QFrame(left);
        previewFrame->setObjectName(QStringLiteral("previewPanel"));
        auto *previewLayout = new QVBoxLayout(previewFrame);
        previewLayout->setContentsMargins(10, 10, 10, 10);

        auto *previewTitle = new QLabel(QStringLiteral("Central Preview (Drag & Drop Supported)"), previewFrame);
        previewTitle->setObjectName(QStringLiteral("panelTitle"));
        previewLayout->addWidget(previewTitle);

        auto *previewInputs = new QGroupBox(QStringLiteral("Preview Sources"), previewFrame);
        auto *previewForm = new QFormLayout(previewInputs);
        m_inputPreviewPath = new QLineEdit(previewInputs);
        auto *browseIn = new QPushButton(QStringLiteral("Browse"), previewInputs);
        m_outputPreviewPath = new QLineEdit(previewInputs);
        auto *browseOut = new QPushButton(QStringLiteral("Browse"), previewInputs);
        previewForm->addRow(QStringLiteral("Input:"), buildPathRow(previewInputs, m_inputPreviewPath, browseIn));
        previewForm->addRow(QStringLiteral("Output:"), buildPathRow(previewInputs, m_outputPreviewPath, browseOut));
        previewLayout->addWidget(previewInputs);

        auto *split = new QSplitter(Qt::Horizontal, previewFrame);
        m_splitInputPreview = new QLabel(QStringLiteral("Input"), split);
        m_splitOutputPreview = new QLabel(QStringLiteral("Output"), split);
        for (auto *lbl : {m_splitInputPreview, m_splitOutputPreview}) {
            lbl->setObjectName(QStringLiteral("previewBox"));
            lbl->setAlignment(Qt::AlignCenter);
            lbl->setMinimumSize(420, 300);
        }

        m_previewTabs = new QTabWidget(previewFrame);
        auto *splitTab = new QWidget(m_previewTabs);
        auto *splitLayout = new QVBoxLayout(splitTab);
        splitLayout->addWidget(split);
        auto *inputTab = new QWidget(m_previewTabs);
        auto *inputLayout = new QVBoxLayout(inputTab);
        m_inputPreviewSolo = new QLabel(QStringLiteral("Input"), inputTab);
        m_inputPreviewSolo->setObjectName(QStringLiteral("previewBox"));
        m_inputPreviewSolo->setAlignment(Qt::AlignCenter);
        inputLayout->addWidget(m_inputPreviewSolo);
        auto *outputTab = new QWidget(m_previewTabs);
        auto *outputLayout = new QVBoxLayout(outputTab);
        m_outputPreviewSolo = new QLabel(QStringLiteral("Output"), outputTab);
        m_outputPreviewSolo->setObjectName(QStringLiteral("previewBox"));
        m_outputPreviewSolo->setAlignment(Qt::AlignCenter);
        outputLayout->addWidget(m_outputPreviewSolo);

        m_previewTabs->addTab(splitTab, QStringLiteral("Split View"));
        m_previewTabs->addTab(inputTab, QStringLiteral("Input"));
        m_previewTabs->addTab(outputTab, QStringLiteral("Output"));
        previewLayout->addWidget(m_previewTabs, 1);

        leftLayout->addWidget(previewFrame, 5);

        auto *playbackFrame = new QFrame(left);
        playbackFrame->setObjectName(QStringLiteral("playbackPanel"));
        auto *playbackLayout = new QHBoxLayout(playbackFrame);
        playbackLayout->setContentsMargins(10, 8, 10, 8);
        m_stepBackBtn = new QPushButton(QStringLiteral("⏮"), playbackFrame);
        m_playBtn = new QPushButton(QStringLiteral("▶"), playbackFrame);
        m_pauseBtn = new QPushButton(QStringLiteral("⏸"), playbackFrame);
        m_stepFwdBtn = new QPushButton(QStringLiteral("⏭"), playbackFrame);
        m_seekSlider = new QSlider(Qt::Horizontal, playbackFrame);
        m_seekSlider->setRange(0, 1000);
        playbackLayout->addWidget(m_stepBackBtn);
        playbackLayout->addWidget(m_playBtn);
        playbackLayout->addWidget(m_pauseBtn);
        playbackLayout->addWidget(m_stepFwdBtn);
        playbackLayout->addWidget(m_seekSlider, 1);
        leftLayout->addWidget(playbackFrame);

        auto *timelineFrame = new QFrame(left);
        timelineFrame->setObjectName(QStringLiteral("timelinePanel"));
        auto *timelineLayout = new QVBoxLayout(timelineFrame);
        timelineLayout->setContentsMargins(10, 8, 10, 8);
        auto *timelineTitle = new QLabel(QStringLiteral("Timeline"), timelineFrame);
        timelineTitle->setObjectName(QStringLiteral("panelTitle"));
        timelineLayout->addWidget(timelineTitle);

        auto *timelineScroll = new QScrollArea(timelineFrame);
        timelineScroll->setWidgetResizable(true);
        timelineScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        timelineScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_timelineHost = new QWidget(timelineScroll);
        m_timelineLayout = new QHBoxLayout(m_timelineHost);
        m_timelineLayout->setContentsMargins(8, 8, 8, 8);
        m_timelineLayout->setSpacing(6);
        timelineScroll->setWidget(m_timelineHost);
        timelineLayout->addWidget(timelineScroll);
        leftLayout->addWidget(timelineFrame);

        auto *progressFrame = new QFrame(left);
        progressFrame->setObjectName(QStringLiteral("progressPanel"));
        auto *prog = new QVBoxLayout(progressFrame);
        prog->setContentsMargins(10, 8, 10, 8);
        m_fileLabel = new QLabel(QStringLiteral("Current File: — (0/0)"), progressFrame);
        m_fileProgress = new QProgressBar(progressFrame);
        m_batchLabel = new QLabel(QStringLiteral("Batch Progress: 0/0"), progressFrame);
        m_batchProgress = new QProgressBar(progressFrame);
        prog->addWidget(m_fileLabel);
        prog->addWidget(m_fileProgress);
        prog->addWidget(m_batchLabel);
        prog->addWidget(m_batchProgress);
        leftLayout->addWidget(progressFrame);

        m_logGroup = new QGroupBox(QStringLiteral("Log Console"), left);
        m_logGroup->setCheckable(true);
        m_logGroup->setChecked(true);
        auto *logLayout = new QVBoxLayout(m_logGroup);
        m_logConsole = new QPlainTextEdit(m_logGroup);
        m_logConsole->setReadOnly(true);
        m_logConsole->setMaximumBlockCount(12000);
        logLayout->addWidget(m_logConsole);
        leftLayout->addWidget(m_logGroup, 2);

        // Right side: collapsible categories
        auto *right = new QFrame(mainSplit);
        right->setObjectName(QStringLiteral("settingsPanel"));
        auto *rightLayout = new QVBoxLayout(right);
        rightLayout->setContentsMargins(10, 10, 10, 10);

        auto *sideTitle = new QLabel(QStringLiteral("Side Panel"), right);
        sideTitle->setObjectName(QStringLiteral("panelTitle"));
        rightLayout->addWidget(sideTitle);

        auto *toolbox = new QToolBox(right);

        QWidget *enhancePage = new QWidget(toolbox);
        auto *enhanceLayout = new QVBoxLayout(enhancePage);

        auto *modelGroup = new QGroupBox(QStringLiteral("Enhancements"), enhancePage);
        auto *mform = new QFormLayout(modelGroup);
        m_ditModelCombo = new QComboBox(modelGroup);
        m_ditModelCombo->addItems({
            QStringLiteral("seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
            QStringLiteral("seedvr2_ema_3b-Q8_0.gguf"),
            QStringLiteral("seedvr2_ema_7b_fp16.safetensors"),
            QStringLiteral("seedvr2_ema_7b_sharp_fp16.safetensors")
        });
        m_preDownscaleCombo = new QComboBox(modelGroup);
        m_preDownscaleCombo->addItems({QStringLiteral("1:1"), QStringLiteral("2:1"), QStringLiteral("3:1")});
        m_resolutionModeCombo = new QComboBox(modelGroup);
        m_resolutionModeCombo->addItems({QStringLiteral("Pixel"), QStringLiteral("X Times")});
        m_resolutionSpin = new QSpinBox(modelGroup);
        m_resolutionSpin->setRange(128, 7680);
        m_resolutionSpin->setValue(720);
        m_resolutionScaleCombo = new QComboBox(modelGroup);
        m_resolutionScaleCombo->addItems({QStringLiteral("1x"), QStringLiteral("2x"), QStringLiteral("3x"), QStringLiteral("4x"), QStringLiteral("5x")});
        m_maxResolutionSpin = new QSpinBox(modelGroup);
        m_maxResolutionSpin->setRange(0, 7680);
        m_batchSizeSpin = new QSpinBox(modelGroup);
        m_batchSizeSpin->setRange(1, 10001);
        m_batchSizeSpin->setValue(81);
        m_uniformBatchCheck = new QCheckBox(modelGroup);
        m_seedSpin = new QSpinBox(modelGroup);
        m_seedSpin->setRange(0, 2147483647);
        m_seedSpin->setValue(313);
        m_skipFirstFramesSpin = new QSpinBox(modelGroup);
        m_skipFirstFramesSpin->setRange(0, 99999);
        m_loadCapSpin = new QSpinBox(modelGroup);
        m_loadCapSpin->setRange(0, 99999);
        m_chunkSizeSpin = new QSpinBox(modelGroup);
        m_chunkSizeSpin->setRange(0, 99999);
        m_prependFramesSpin = new QSpinBox(modelGroup);
        m_prependFramesSpin->setRange(0, 100);
        m_temporalOverlapSpin = new QSpinBox(modelGroup);
        m_temporalOverlapSpin->setRange(0, 100);
        m_colorCorrectionCombo = new QComboBox(modelGroup);
        m_colorCorrectionCombo->addItems({QStringLiteral("lab"), QStringLiteral("wavelet"), QStringLiteral("wavelet_adaptive"), QStringLiteral("hsv"), QStringLiteral("adain"), QStringLiteral("none")});
        m_attentionModeCombo = new QComboBox(modelGroup);
        m_attentionModeCombo->addItems({QStringLiteral("sdpa"), QStringLiteral("flash_attn_2"), QStringLiteral("flash_attn_3"), QStringLiteral("sageattn_2"), QStringLiteral("sageattn_3")});

        m_ditOffloadCombo = new QComboBox(modelGroup);
        m_ditOffloadCombo->addItems({QStringLiteral("none"), QStringLiteral("cpu"), QStringLiteral("0"), QStringLiteral("1")});
        m_vaeOffloadCombo = new QComboBox(modelGroup);
        m_vaeOffloadCombo->addItems({QStringLiteral("none"), QStringLiteral("cpu"), QStringLiteral("0"), QStringLiteral("1")});
        m_tensorOffloadCombo = new QComboBox(modelGroup);
        m_tensorOffloadCombo->addItems({QStringLiteral("cpu"), QStringLiteral("none"), QStringLiteral("0"), QStringLiteral("1")});

        m_blocksToSwapSpin = new QSpinBox(modelGroup);
        m_blocksToSwapSpin->setRange(0, 64);
        m_swapIoCheck = new QCheckBox(modelGroup);
        m_vaeEncodeTiledCheck = new QCheckBox(modelGroup);
        m_vaeEncodeTileSizeSpin = new QSpinBox(modelGroup);
        m_vaeEncodeTileSizeSpin->setRange(128, 4096);
        m_vaeEncodeTileSizeSpin->setValue(1024);
        m_vaeEncodeTileOverlapSpin = new QSpinBox(modelGroup);
        m_vaeEncodeTileOverlapSpin->setRange(0, 1023);
        m_vaeEncodeTileOverlapSpin->setValue(128);
        m_vaeDecodeTiledCheck = new QCheckBox(modelGroup);
        m_vaeDecodeTileSizeSpin = new QSpinBox(modelGroup);
        m_vaeDecodeTileSizeSpin->setRange(128, 4096);
        m_vaeDecodeTileSizeSpin->setValue(1024);
        m_vaeDecodeTileOverlapSpin = new QSpinBox(modelGroup);
        m_vaeDecodeTileOverlapSpin->setRange(0, 1023);
        m_vaeDecodeTileOverlapSpin->setValue(128);
        m_tileDebugCombo = new QComboBox(modelGroup);
        m_tileDebugCombo->addItems({QStringLiteral("false"), QStringLiteral("encode"), QStringLiteral("decode")});
        m_cacheDitCheck = new QCheckBox(modelGroup);
        m_cacheVaeCheck = new QCheckBox(modelGroup);
        m_autoSafeguardCheck = new QCheckBox(modelGroup);
        m_debugCheck = new QCheckBox(modelGroup);

        mform->addRow(QStringLiteral("DiT Model:"), m_ditModelCombo);
        mform->addRow(QStringLiteral("Pre-Downscale:"), m_preDownscaleCombo);
        mform->addRow(QStringLiteral("Resolution Mode:"), m_resolutionModeCombo);
        mform->addRow(QStringLiteral("Resolution Pixel:"), m_resolutionSpin);
        mform->addRow(QStringLiteral("Resolution Scale:"), m_resolutionScaleCombo);
        mform->addRow(QStringLiteral("Max Resolution:"), m_maxResolutionSpin);
        mform->addRow(QStringLiteral("Batch Size:"), m_batchSizeSpin);
        mform->addRow(QStringLiteral("Uniform Batch:"), m_uniformBatchCheck);
        mform->addRow(QStringLiteral("Seed:"), m_seedSpin);
        mform->addRow(QStringLiteral("Skip First Frames:"), m_skipFirstFramesSpin);
        mform->addRow(QStringLiteral("Load Cap:"), m_loadCapSpin);
        mform->addRow(QStringLiteral("Chunk Size:"), m_chunkSizeSpin);
        mform->addRow(QStringLiteral("Prepend Frames:"), m_prependFramesSpin);
        mform->addRow(QStringLiteral("Temporal Overlap:"), m_temporalOverlapSpin);
        mform->addRow(QStringLiteral("Color Correction:"), m_colorCorrectionCombo);
        mform->addRow(QStringLiteral("Attention Mode:"), m_attentionModeCombo);
        mform->addRow(QStringLiteral("DiT Offload:"), m_ditOffloadCombo);
        mform->addRow(QStringLiteral("VAE Offload:"), m_vaeOffloadCombo);
        mform->addRow(QStringLiteral("Tensor Offload:"), m_tensorOffloadCombo);
        mform->addRow(QStringLiteral("Blocks To Swap:"), m_blocksToSwapSpin);
        mform->addRow(QStringLiteral("Swap IO Components:"), m_swapIoCheck);
        mform->addRow(QStringLiteral("VAE Encode Tiled:"), m_vaeEncodeTiledCheck);
        mform->addRow(QStringLiteral("Encode Tile Size:"), m_vaeEncodeTileSizeSpin);
        mform->addRow(QStringLiteral("Encode Tile Overlap:"), m_vaeEncodeTileOverlapSpin);
        mform->addRow(QStringLiteral("VAE Decode Tiled:"), m_vaeDecodeTiledCheck);
        mform->addRow(QStringLiteral("Decode Tile Size:"), m_vaeDecodeTileSizeSpin);
        mform->addRow(QStringLiteral("Decode Tile Overlap:"), m_vaeDecodeTileOverlapSpin);
        mform->addRow(QStringLiteral("Tile Debug:"), m_tileDebugCombo);
        mform->addRow(QStringLiteral("Cache DiT:"), m_cacheDitCheck);
        mform->addRow(QStringLiteral("Cache VAE:"), m_cacheVaeCheck);
        mform->addRow(QStringLiteral("Auto Safeguard:"), m_autoSafeguardCheck);
        mform->addRow(QStringLiteral("Debug:"), m_debugCheck);

        enhanceLayout->addWidget(modelGroup);
        enhanceLayout->addStretch(1);

        QWidget *codecPage = new QWidget(toolbox);
        auto *codecLayout = new QVBoxLayout(codecPage);
        auto *codecGroup = new QGroupBox(QStringLiteral("Codec Settings"), codecPage);
        auto *cform = new QFormLayout(codecGroup);
        m_outputTypeCombo = new QComboBox(codecGroup);
        m_outputTypeCombo->addItems({QStringLiteral("Video"), QStringLiteral("Image Sequence")});
        m_codecCombo = new QComboBox(codecGroup);
        m_codecCombo->addItems({QStringLiteral("H265"), QStringLiteral("H264")});
        m_bitrateModeCombo = new QComboBox(codecGroup);
        m_bitrateModeCombo->addItems({QStringLiteral("Dynamic"), QStringLiteral("Constant")});
        m_qualityCombo = new QComboBox(codecGroup);
        m_qualityCombo->addItems({QStringLiteral("Low"), QStringLiteral("Medium"), QStringLiteral("High")});
        m_containerCombo = new QComboBox(codecGroup);
        m_containerCombo->addItems({QStringLiteral("mp4"), QStringLiteral("mkv")});
        m_lutPathEdit = new QLineEdit(codecGroup);
        auto *lutBrowse = new QPushButton(QStringLiteral("Browse"), codecGroup);
        cform->addRow(QStringLiteral("Output Type:"), m_outputTypeCombo);
        cform->addRow(QStringLiteral("Codec:"), m_codecCombo);
        cform->addRow(QStringLiteral("Bitrate:"), m_bitrateModeCombo);
        cform->addRow(QStringLiteral("Quality:"), m_qualityCombo);
        cform->addRow(QStringLiteral("Container:"), m_containerCombo);
        cform->addRow(QStringLiteral("LUT:"), buildPathRow(codecGroup, m_lutPathEdit, lutBrowse));
        codecLayout->addWidget(codecGroup);
        codecLayout->addStretch(1);

        QWidget *presetsPage = new QWidget(toolbox);
        auto *presetsLayout = new QVBoxLayout(presetsPage);
        auto *savePresetBtn = new QPushButton(QStringLiteral("Save Preset..."), presetsPage);
        auto *loadPresetBtn = new QPushButton(QStringLiteral("Load Preset..."), presetsPage);
        presetsLayout->addWidget(savePresetBtn);
        presetsLayout->addWidget(loadPresetBtn);
        presetsLayout->addStretch(1);

        toolbox->addItem(enhancePage, QStringLiteral("Enhancements"));
        toolbox->addItem(codecPage, QStringLiteral("Codec Settings"));
        toolbox->addItem(presetsPage, QStringLiteral("Presets"));
        rightLayout->addWidget(toolbox, 1);

        auto *btnRow = new QHBoxLayout();
        m_startBtn = new QPushButton(QStringLiteral("Start"), right);
        m_startBtn->setObjectName(QStringLiteral("startButton"));
        m_stopBtn = new QPushButton(QStringLiteral("Stop"), right);
        m_stopBtn->setObjectName(QStringLiteral("stopButton"));
        btnRow->addWidget(m_startBtn);
        btnRow->addWidget(m_stopBtn);
        rightLayout->addLayout(btnRow);

        mainSplit->addWidget(left);
        mainSplit->addWidget(right);
        mainSplit->setStretchFactor(0, 7);
        mainSplit->setStretchFactor(1, 3);

        connect(browseIn, &QPushButton::clicked, this, &MainWindow::browsePreviewInput);
        connect(browseOut, &QPushButton::clicked, this, &MainWindow::browsePreviewOutput);
        connect(lutBrowse, &QPushButton::clicked, this, &MainWindow::browseLutPath);
        connect(m_resolutionModeCombo, &QComboBox::currentTextChanged, this, &MainWindow::updateResolutionMode);
        connect(m_startBtn, &QPushButton::clicked, this, &MainWindow::startRender);
        connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::stopRender);
        connect(savePresetBtn, &QPushButton::clicked, this, &MainWindow::savePreset);
        connect(loadPresetBtn, &QPushButton::clicked, this, &MainWindow::loadPreset);
        connect(m_logGroup, &QGroupBox::toggled, this, [this](bool on) { m_logConsole->setVisible(on); });

        updateResolutionMode(m_resolutionModeCombo->currentText());
        rebuildTimelineMarkers();
        refreshPreviewPanels();
        statusBar()->showMessage(QStringLiteral("Ready"));
    }

    void rebuildTimelineMarkers()
    {
        while (QLayoutItem *item = m_timelineLayout->takeAt(0)) {
            if (item->widget()) item->widget()->deleteLater();
            delete item;
        }

        for (int i = 0; i < 24; ++i) {
            auto *thumb = new QFrame(m_timelineHost);
            thumb->setObjectName(QStringLiteral("card"));
            thumb->setFixedSize(68, 50);
            auto *lay = new QVBoxLayout(thumb);
            lay->setContentsMargins(4, 4, 4, 4);
            auto *lbl = new QLabel(QStringLiteral("%1:%2").arg(i / 2, 2, 10, QLatin1Char('0')).arg((i % 2) * 30, 2, 10, QLatin1Char('0')), thumb);
            lbl->setAlignment(Qt::AlignCenter);
            lay->addWidget(lbl);
            m_timelineLayout->addWidget(thumb);
        }
        m_timelineLayout->addStretch(1);
    }

    void saveUiSettings()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        s.setValue(QStringLiteral("ui/inputPreview"), m_inputPreviewPath->text());
        s.setValue(QStringLiteral("ui/outputPreview"), m_outputPreviewPath->text());
        s.setValue(QStringLiteral("ui/dmodel"), m_ditModelCombo->currentText());
        s.setValue(QStringLiteral("ui/outputType"), m_outputTypeCombo->currentText());
        s.setValue(QStringLiteral("ui/codec"), m_codecCombo->currentText());
        s.setValue(QStringLiteral("ui/container"), m_containerCombo->currentText());
        s.setValue(QStringLiteral("ui/lut"), m_lutPathEdit->text());
    }

    void loadUiSettings()
    {
        QSettings s(QStringLiteral("SeedVR2"), QStringLiteral("QtRunner"));
        m_inputPreviewPath->setText(s.value(QStringLiteral("ui/inputPreview")).toString());
        m_outputPreviewPath->setText(s.value(QStringLiteral("ui/outputPreview")).toString());
        m_ditModelCombo->setCurrentText(s.value(QStringLiteral("ui/dmodel"), m_ditModelCombo->currentText()).toString());
        m_outputTypeCombo->setCurrentText(s.value(QStringLiteral("ui/outputType"), QStringLiteral("Video")).toString());
        m_codecCombo->setCurrentText(s.value(QStringLiteral("ui/codec"), QStringLiteral("H265")).toString());
        m_containerCombo->setCurrentText(s.value(QStringLiteral("ui/container"), QStringLiteral("mp4")).toString());
        m_lutPathEdit->setText(s.value(QStringLiteral("ui/lut")).toString());
    }

    // core runtime objects
    ProcessEngine *m_engine = nullptr;
    SettingsDialog *m_settingsDialog = nullptr;

    // preview
    QTabWidget *m_previewTabs = nullptr;
    QLabel *m_splitInputPreview = nullptr;
    QLabel *m_splitOutputPreview = nullptr;
    QLabel *m_inputPreviewSolo = nullptr;
    QLabel *m_outputPreviewSolo = nullptr;
    QLineEdit *m_inputPreviewPath = nullptr;
    QLineEdit *m_outputPreviewPath = nullptr;

    // timeline/playback
    QPushButton *m_stepBackBtn = nullptr;
    QPushButton *m_playBtn = nullptr;
    QPushButton *m_pauseBtn = nullptr;
    QPushButton *m_stepFwdBtn = nullptr;
    QSlider *m_seekSlider = nullptr;
    QWidget *m_timelineHost = nullptr;
    QHBoxLayout *m_timelineLayout = nullptr;

    // side controls
    QComboBox *m_ditModelCombo = nullptr;
    QComboBox *m_preDownscaleCombo = nullptr;
    QComboBox *m_resolutionModeCombo = nullptr;
    QSpinBox *m_resolutionSpin = nullptr;
    QComboBox *m_resolutionScaleCombo = nullptr;
    QSpinBox *m_maxResolutionSpin = nullptr;
    QSpinBox *m_batchSizeSpin = nullptr;
    QCheckBox *m_uniformBatchCheck = nullptr;
    QSpinBox *m_seedSpin = nullptr;
    QSpinBox *m_skipFirstFramesSpin = nullptr;
    QSpinBox *m_loadCapSpin = nullptr;
    QSpinBox *m_chunkSizeSpin = nullptr;
    QSpinBox *m_prependFramesSpin = nullptr;
    QSpinBox *m_temporalOverlapSpin = nullptr;
    QComboBox *m_colorCorrectionCombo = nullptr;
    QComboBox *m_attentionModeCombo = nullptr;
    QComboBox *m_ditOffloadCombo = nullptr;
    QComboBox *m_vaeOffloadCombo = nullptr;
    QComboBox *m_tensorOffloadCombo = nullptr;
    QSpinBox *m_blocksToSwapSpin = nullptr;
    QCheckBox *m_swapIoCheck = nullptr;
    QCheckBox *m_vaeEncodeTiledCheck = nullptr;
    QSpinBox *m_vaeEncodeTileSizeSpin = nullptr;
    QSpinBox *m_vaeEncodeTileOverlapSpin = nullptr;
    QCheckBox *m_vaeDecodeTiledCheck = nullptr;
    QSpinBox *m_vaeDecodeTileSizeSpin = nullptr;
    QSpinBox *m_vaeDecodeTileOverlapSpin = nullptr;
    QComboBox *m_tileDebugCombo = nullptr;
    QCheckBox *m_cacheDitCheck = nullptr;
    QCheckBox *m_cacheVaeCheck = nullptr;
    QCheckBox *m_autoSafeguardCheck = nullptr;
    QCheckBox *m_debugCheck = nullptr;

    // codec panel
    QComboBox *m_outputTypeCombo = nullptr;
    QComboBox *m_codecCombo = nullptr;
    QComboBox *m_bitrateModeCombo = nullptr;
    QComboBox *m_qualityCombo = nullptr;
    QComboBox *m_containerCombo = nullptr;
    QLineEdit *m_lutPathEdit = nullptr;

    // run/progress/log
    QPushButton *m_startBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    QLabel *m_fileLabel = nullptr;
    QLabel *m_batchLabel = nullptr;
    QProgressBar *m_fileProgress = nullptr;
    QProgressBar *m_batchProgress = nullptr;
    QGroupBox *m_logGroup = nullptr;
    QPlainTextEdit *m_logConsole = nullptr;
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
