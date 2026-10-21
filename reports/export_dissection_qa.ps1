$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$inputFile = Join-Path $reportRoot '拆解报告_GAS_MARL.docx'
$qaRoot = Join-Path $reportRoot 'qa_dissection'
[System.IO.Directory]::CreateDirectory($qaRoot) | Out-Null
$pdfFile = Join-Path $qaRoot 'dissection.pdf'
$wordInstance = $null
$ownedDocument = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $wordInstance.AutomationSecurity = 3
    $ownedDocument = $wordInstance.Documents.Open($inputFile, $false, $true, $false)
    $ownedDocument.ExportAsFixedFormat($pdfFile, 17)
    Write-Output $pdfFile
} finally {
    if ($null -ne $ownedDocument) {
        $ownedDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ownedDocument)
    }
    if ($null -ne $wordInstance) {
        $wordInstance.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordInstance)
    }
}
