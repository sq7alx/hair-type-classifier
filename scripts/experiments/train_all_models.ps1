    $ErrorActionPreference = "Stop"

    $RUN_ID = Get-Date -Format "yyyyMMdd_HHmmss"

    $LOG_DIR = "logs\train\$RUN_ID"
    $CHECKPOINT_DIR = "checkpoints\$RUN_ID"
    New-Item -ItemType Directory -Force -Path $LOG_DIR, $CHECKPOINT_DIR | Out-Null

    # parameters
    $ARCH = "resnet18"
    $EPOCHS = 100
    $LR = "0.0001"
    $SEED = 1234
    $PATIENCE = 20
    $DEVICE = "cuda"

    Write-Host "Experiment ID: $RUN_ID"
    Write-Host "Logs directory: $LOG_DIR"
    Write-Host "Checkpoints directory: $CHECKPOINT_DIR"

    # training function
    function Run-Training($desc, $parentClass, $outDir, $logFile) {
        Write-Host "------------------------------------------------------" -ForegroundColor Cyan
        Write-Host $desc -ForegroundColor Green

        $pyArgs = @(
            "src/train/train.py",
            "--arch", $ARCH,
            "--epochs", $EPOCHS,
            "--lr", $LR,
            "--seed", $SEED,
            "--patience", $PATIENCE,
            "--device", $DEVICE,
            "--output", $outDir
        )

        if ($parentClass -ne $null) {
            $pyArgs += @("--parent-class", $parentClass)
        }

        try {
            cmd /c "python $($pyArgs -join ' ') 2>&1" | Tee-Object -FilePath "$logFile.log"
        } catch {
            # ignore PowerShell NativeCommandError from stderr (tqdm / logging)
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed with exit code $LASTEXITCODE. Check log: $logFile.log"
        }
    }

    #  training H0, H1, H2, H3

    # H0 - main
    Run-Training "[1/4] Training H0 (Main)" $null "$CHECKPOINT_DIR\h0_main" "$LOG_DIR\h0_main.log"

    # H1 - straight
    Run-Training "[2/4] Training H1 (Straight)" 1 "$CHECKPOINT_DIR\h1_straight" "$LOG_DIR\h1_straight.log"

    # H2 - wavy
    Run-Training "[3/4] Training H2 (Wavy)" 2 "$CHECKPOINT_DIR\h2_wavy" "$LOG_DIR\h2_wavy.log"

    # H3 - curly
    Run-Training "[4/4] Training H3 (Curly)" 3 "$CHECKPOINT_DIR\h3_curly" "$LOG_DIR\h3_curly.log"

    Write-Host "All trainings finished successfully."
    Write-Host "Logs: $LOG_DIR"
    Write-Host "Checkpoints: $CHECKPOINT_DIR"
