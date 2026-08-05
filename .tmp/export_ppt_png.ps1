param(
  [Parameter(Mandatory = $true)]
  [string]$DeckPath,
  [Parameter(Mandatory = $true)]
  [string]$OutputDirectory
)

$resolvedDeckPath = (Resolve-Path -LiteralPath $DeckPath).Path
$resolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($resolvedOutputDirectory) | Out-Null

$powerPoint = $null
$presentation = $null
$createdApplication = $false

try {
  try {
    $powerPoint = [Runtime.InteropServices.Marshal]::GetActiveObject(
      'PowerPoint.Application'
    )
  }
  catch {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $createdApplication = $true
  }

  $presentation = $powerPoint.Presentations.Open(
    $resolvedDeckPath,
    -1,
    0,
    0
  )

  foreach ($slide in $presentation.Slides) {
    $outputPath = Join-Path (
      $resolvedOutputDirectory
    ) ("slide-{0}.png" -f $slide.SlideIndex)
    $slide.Export($outputPath, 'PNG', 1920, 1080)
  }
}
finally {
  if ($null -ne $presentation) {
    $presentation.Close()
  }
  if ($createdApplication -and $null -ne $powerPoint) {
    $powerPoint.Quit()
  }
  if ($null -ne $presentation) {
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
  }
  if ($null -ne $powerPoint) {
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
  }
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}
