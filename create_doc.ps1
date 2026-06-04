
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Add()
    $selection = $word.Selection

    # Title
    $selection.Font.Size = 26
    $selection.Font.Bold = $true
    $selection.ParagraphFormat.Alignment = 1 # Center
    $selection.TypeText("StockIntel - Professional Analysis Report (v2.0)`r`n")
    $selection.ParagraphFormat.Alignment = 0 # Left
    
    # Overview
    $selection.Font.Size = 14
    $selection.Font.Bold = $true
    $selection.TypeText("`r`nProject Overview`r`n")
    $selection.Font.Size = 11
    $selection.Font.Bold = $false
    $selection.TypeText("StockIntel is a high-performance financial intelligence dashboard. Version 2.0 introduces deep stability fixes, instant-load architecture, and a modular UI designed for professional stock analysis in the Indian market.`r`n")

    # New Stability Features (v2.0)
    $selection.Font.Size = 14
    $selection.Font.Bold = $true
    $selection.TypeText("`r`nNew Features & Stability (v2.0)`r`n")
    $selection.Font.Size = 11
    $selection.Font.Bold = $false
    $selection.TypeText("1. Instant-Load Architecture: Implemented 'Lazy-Loading' for heavy AI models (FinBERT/TensorFlow). The homepage now loads in milliseconds.`r`n")
    $selection.TypeText("2. Stable Port Management: Migrated to Port 5060 to ensure zero conflicts with local system services.`r`n")
    $selection.TypeText("3. Full Detail Pages: Added dedicated analytics views for Detailed Sentiment, Sector Intelligence, and Final Signal Reports.`r`n")
    $selection.TypeText("4. Public Data Scraper: Bypassed API restrictions by implementing a Public JSON Scraper for Reddit, removing the need for Client IDs.`r`n")
    $selection.TypeText("5. AI Sanity Constraints: Added a safety layer to the LSTM model to prevent unrealistic price target outliers.`r`n")
    $selection.TypeText("6. Brand Evolution: Unified the application under the 'StockIntel' identity with professional headers and reporting logic.`r`n")

    # Technical Architecture
    $selection.Font.Size = 14
    $selection.Font.Bold = $true
    $selection.TypeText("`r`nTechnical Core`r`n")
    $selection.Font.Size = 11
    $selection.Font.Bold = $false
    $selection.TypeText("- ML Predictor: Hybrid LSTM (Pattern Recognition) + XGBoost (Probability Modeling).`r`n")
    $selection.TypeText("- Sentiment Engine: ProsusAI/FinBERT Transformer for high-signal financial NLP.`r`n")
    $selection.TypeText("- Data Controller: Automated NSE Pipeline via yfinance with multi-year historical depth.`r`n")
    $selection.TypeText("- Signal Aggregator: Weighted logic (30% Price, 20% News, 15% Social, 20% Macro, 15% Sector).`r`n")

    $filename = "C:\Users\abhyu\OneDrive\Desktop\Stocksense\StockIntel_Final_Report_v2.docx"
    $doc.SaveAs([ref]$filename)
    $doc.Close()
    $word.Quit()
    Write-Host "Success: Final Report saved to $filename"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($word) { $word.Quit() }
}
