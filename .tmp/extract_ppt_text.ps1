param(
  [Parameter(Mandatory = $true)]
  [string]$DeckPath
)

$resolvedDeckPath = (Resolve-Path -LiteralPath $DeckPath).Path
Add-Type -AssemblyName System.IO.Compression

$stream = [System.IO.FileStream]::new(
  $resolvedDeckPath,
  [System.IO.FileMode]::Open,
  [System.IO.FileAccess]::Read,
  [System.IO.FileShare]::ReadWrite
)
$zip = [System.IO.Compression.ZipArchive]::new(
  $stream,
  [System.IO.Compression.ZipArchiveMode]::Read,
  $false
)

try {
  $slides = $zip.Entries |
    Where-Object { $_.FullName -match '^ppt/slides/slide(\d+)\.xml$' } |
    Sort-Object { [int]([regex]::Match($_.FullName, 'slide(\d+)').Groups[1].Value) }

  foreach ($entry in $slides) {
    $slideNumber = [int]([regex]::Match($entry.FullName, 'slide(\d+)').Groups[1].Value)
    $reader = [System.IO.StreamReader]::new($entry.Open())
    try {
      [xml]$xml = $reader.ReadToEnd()
    }
    finally {
      $reader.Dispose()
    }

    $namespaceManager = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace(
      'a',
      'http://schemas.openxmlformats.org/drawingml/2006/main'
    )
    $texts = @(
      $xml.SelectNodes('//a:t', $namespaceManager) |
        ForEach-Object { $_.'#text' }
    )

    Write-Output "--- SLIDE $slideNumber ---"
    Write-Output ($texts -join ' | ')

    $relsEntry = $zip.GetEntry(
      "ppt/slides/_rels/slide$slideNumber.xml.rels"
    )
    if ($null -ne $relsEntry) {
      $relsReader = [System.IO.StreamReader]::new($relsEntry.Open())
      try {
        [xml]$relsXml = $relsReader.ReadToEnd()
      }
      finally {
        $relsReader.Dispose()
      }
      $notesRelationship = $relsXml.Relationships.Relationship |
        Where-Object {
          $_.Type -eq (
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide'
          )
        } |
        Select-Object -First 1
      if ($null -ne $notesRelationship) {
        $notesTarget = $notesRelationship.Target -replace '^\.\./', 'ppt/'
        $notesEntry = $zip.GetEntry($notesTarget)
        if ($null -ne $notesEntry) {
          $notesReader = [System.IO.StreamReader]::new($notesEntry.Open())
          try {
            [xml]$notesXml = $notesReader.ReadToEnd()
          }
          finally {
            $notesReader.Dispose()
          }
          $notesNamespaceManager = [System.Xml.XmlNamespaceManager]::new(
            $notesXml.NameTable
          )
          $notesNamespaceManager.AddNamespace(
            'a',
            'http://schemas.openxmlformats.org/drawingml/2006/main'
          )
          $notesTexts = @(
            $notesXml.SelectNodes('//a:t', $notesNamespaceManager) |
              ForEach-Object { $_.'#text' }
          )
          if ($notesTexts.Count -gt 0) {
            Write-Output "--- NOTES $slideNumber ---"
            Write-Output ($notesTexts -join ' | ')
          }
        }
      }
    }
  }
}
finally {
  $zip.Dispose()
  $stream.Dispose()
}
