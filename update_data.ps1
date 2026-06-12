Write-Host "========================================"
Write-Host "StockSense India - Automated Data Update"
Write-Host "========================================"
Write-Host "Step 1/3: Training AI Models (This will take a few minutes)..."
python batch_train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during training. Aborting update." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Step 2/3: Running Execution Pipeline (Fetching data & predicting)..."
python batch_runner.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during execution. Aborting update." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Step 3/3: Committing and Pushing to GitHub..."
git add data/ models/
git commit -m "Automated local update: Refreshed AI models and stock data"

# Git commit returns 1 if there are no changes, so we don't abort on 1.
git push origin master

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during git push. Please check your connection." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "========================================"
Write-Host "Update Complete! Render will deploy the changes shortly." -ForegroundColor Green
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
