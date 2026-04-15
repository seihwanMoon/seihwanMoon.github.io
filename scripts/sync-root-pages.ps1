param(
    [string]$VaultPath = 'G:\My Drive\Bobsidian\party',
    [string]$ExportPath = 'G:\My Drive\Bobsidian\seihwanMoon.github.io'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$templatePath = Join-Path $ExportPath 'wiki\home.html'
$fileTreePath = Join-Path $ExportPath 'site-lib\html\file-tree-content.html'

if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Template page not found: $templatePath"
}

$template = Get-Content -LiteralPath $templatePath -Raw

$linkMap = @{
    'index' = 'index.html'
    'log' = 'log.html'
    'AGENTS' = 'agents.html'
    'Home' = 'wiki/home.html'
    '사용 방법' = 'wiki/사용-방법.html'
    'LLM-Maintained Wiki Pattern' = 'wiki/llm-maintained-wiki-pattern.html'
    'Source - Karpathy - LLM Wiki' = 'wiki/source-karpathy-llm-wiki.html'
}

$pageSpecs = @(
    @{
        Source = 'index.md'
        Output = 'index.html'
        Title = 'Party Wiki 인덱스'
    },
    @{
        Source = 'AGENTS.md'
        Output = 'agents.html'
        Title = 'Party LLM Wiki 운영 규칙'
    },
    @{
        Source = 'log.md'
        Output = 'log.html'
        Title = 'Party Wiki 로그'
    }
)

function Convert-WikiLinks {
    param(
        [string]$Markdown,
        [hashtable]$Map
    )

    return [regex]::Replace(
        $Markdown,
        '\[\[([^\]|]+)(?:\|([^\]]+))?\]\]',
        {
            param($match)

            $target = $match.Groups[1].Value.Trim()
            $label = if ($match.Groups[2].Success) {
                $match.Groups[2].Value.Trim()
            }
            else {
                $target
            }

            if ($Map.ContainsKey($target)) {
                return "[{0}]({1})" -f $label, $Map[$target]
            }

            return $label
        }
    )
}

function Get-MarkdownDescription {
    param([string]$Markdown)

    foreach ($line in ($Markdown -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed) {
            continue
        }

        if ($trimmed.StartsWith('#') -or $trimmed.StartsWith('```')) {
            continue
        }

        $plain = $trimmed `
            -replace '\[\[([^\]|]+)\|([^\]]+)\]\]', '$2' `
            -replace '\[\[([^\]]+)\]\]', '$1' `
            -replace '`', ''

        if ($plain.Length -gt 160) {
            return $plain.Substring(0, 160)
        }

        return $plain
    }

    return ''
}

function Convert-MarkdownBody {
    param([string]$Markdown)

    $bodyHtml = (ConvertFrom-Markdown -InputObject $Markdown).Html.Trim()
    $headings = [System.Collections.Generic.List[object]]::new()
    $headingState = [pscustomobject]@{
        Index = 0
    }

    $bodyHtml = [regex]::Replace(
        $bodyHtml,
        '<h([1-6]) id="[^"]*">(.*?)</h\1>',
        {
            param($match)

            $headingState.Index += 1
            $level = [int]$match.Groups[1].Value
            $innerHtml = $match.Groups[2].Value
            $text = ([regex]::Replace($innerHtml, '<[^>]+>', '')).Trim()
            $id = "heading-$($headingState.Index)"
            $safeText = [System.Net.WebUtility]::HtmlEncode($text)

            $headings.Add([pscustomobject]@{
                    Id    = $id
                    Level = $level
                    Text  = $text
                })

            return "<div class=`"el-h$level`"><h$level data-heading=`"$safeText`" dir=`"auto`" class=`"heading`" id=`"$id`"><span class=`"heading-collapse-indicator collapse-indicator collapse-icon`"><svg xmlns=`"http://www.w3.org/2000/svg`" width=`"24`" height=`"24`" viewBox=`"0 0 24 24`" fill=`"none`" stroke=`"currentColor`" stroke-width=`"2`" stroke-linecap=`"round`" stroke-linejoin=`"round`" class=`"svg-icon right-triangle`"><path d=`"M3 8L12 17L21 8`"></path></svg></span>$innerHtml</h$level></div>"
        },
        'Singleline'
    )

    return [pscustomobject]@{
        Html     = $bodyHtml
        Headings = $headings
    }
}

function New-OutlineHtml {
    param(
        [string]$PagePath,
        [System.Collections.Generic.List[object]]$Headings
    )

    $buttonSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></svg>'
    $items = foreach ($heading in $Headings) {
        $safeText = [System.Net.WebUtility]::HtmlEncode($heading.Text)
        @"
<div class="tree-item" data-depth="$($heading.Level)"><a class="tree-item-self is-clickable" href="$PagePath#$($heading.Id)" data-path="#$($heading.Id)"><div class="tree-item-inner heading-link" heading-name="$safeText">$safeText</div></a><div class="tree-item-children"></div></div>
"@
    }

    $itemHtml = ($items -join '')

    @"
<div id="outline" class=" tree-container"><div class="feature-header"><div class="feature-title">Table Of Contents</div><button class="clickable-icon nav-action-button tree-collapse-all" aria-label="Collapse All">$buttonSvg</button></div>$itemHtml</div>
"@
}

function New-ContentBlock {
    param(
        [string]$Title,
        [string]$BodyHtml
    )

    $safeTitle = [System.Net.WebUtility]::HtmlEncode($Title)

    @"
<div class="markdown-preview-sizer markdown-preview-section"><div class="header"><h1 class="page-title heading inline-title" id="page-title">$safeTitle</h1><div class="data-bar"></div></div><div class="markdown-preview-pusher" style="width: 1px; height: 0.1px; margin-bottom: 0px;"></div>$BodyHtml<div class="footer"><div class="data-bar"></div></div></div></div></div>
"@
}

function Apply-PageTemplate {
    param(
        [string]$Template,
        [string]$OutputPath,
        [string]$Title,
        [string]$Description,
        [string]$ContentBlock,
        [string]$OutlineHtml
    )

    $safeTitle = [System.Net.WebUtility]::HtmlEncode($Title)
    $safeDescription = [System.Net.WebUtility]::HtmlEncode($Description)
    $safeOutputPath = [System.Net.WebUtility]::HtmlEncode($OutputPath)

    $page = $Template `
        -replace '<title>.*?</title>', "<title>$safeTitle</title>" `
        -replace '<base href="\.\.">', '<base href=".">' `
        -replace '<meta name="pathname" content="[^"]+">', "<meta name=`"pathname`" content=`"$safeOutputPath`">" `
        -replace '<meta name="description" content="[^"]*">', "<meta name=`"description`" content=`"$safeDescription`">" `
        -replace '<meta property="og:title" content="[^"]*">', "<meta property=`"og:title`" content=`"$safeTitle`">" `
        -replace '<meta property="og:description" content="[^"]*">', "<meta property=`"og:description`" content=`"$safeDescription`">" `
        -replace '<meta property="og:url" content="[^"]*">', "<meta property=`"og:url`" content=`"$safeOutputPath`">"

    $centerStartMarker = '<div class="markdown-preview-sizer markdown-preview-section">'
    $rightContentMarker = '<div id="right-content" class="leaf"'
    $centerStart = $page.IndexOf($centerStartMarker)
    $rightContentStart = $page.IndexOf($rightContentMarker)

    if ($centerStart -lt 0 -or $rightContentStart -lt 0) {
        throw 'Could not locate center content markers in template.'
    }

    $page = $page.Substring(0, $centerStart) + $ContentBlock + $page.Substring($rightContentStart)

    $outlineStartMarker = '<div id="outline" class=" tree-container">'
    $outlineEndMarker = '</div></div></div><script defer="">let rs'
    $outlineStart = $page.IndexOf($outlineStartMarker)
    $outlineEnd = $page.IndexOf($outlineEndMarker, $outlineStart)

    if ($outlineStart -lt 0 -or $outlineEnd -lt 0) {
        throw 'Could not locate outline markers in template.'
    }

    $page = $page.Substring(0, $outlineStart) + $OutlineHtml + $page.Substring($outlineEnd)
    return $page
}

function Update-FileTreeRootEntries {
    param([string]$Path)

    $fileTree = Get-Content -LiteralPath $Path -Raw
    $fileTree = $fileTree -replace '<div class="tree-item mod-collapsible" is-collapsed nav-folder" data-depth="1">', '<div class="tree-item mod-collapsible is-collapsed nav-folder" data-depth="1">'

    $rootEntries = @"
<div class="tree-item is-collapsed nav-file" data-depth="1"><a class="tree-item-self is-clickable nav-file-title" href="index.html" data-path="index.md"><div class="tree-item-inner nav-file-title-content">index</div></a><div class="tree-item-children nav-file-children"></div></div><div class="tree-item is-collapsed nav-file" data-depth="1"><a class="tree-item-self is-clickable nav-file-title" href="log.html" data-path="log.md"><div class="tree-item-inner nav-file-title-content">log</div></a><div class="tree-item-children nav-file-children"></div></div><div class="tree-item is-collapsed nav-file" data-depth="1"><a class="tree-item-self is-clickable nav-file-title" href="agents.html" data-path="AGENTS.md"><div class="tree-item-inner nav-file-title-content">AGENTS</div></a><div class="tree-item-children nav-file-children"></div></div>
"@

    $folderMarker = '<div class="tree-item mod-collapsible is-collapsed nav-folder" data-depth="1">'
    $folderIndex = $fileTree.IndexOf($folderMarker)
    if ($folderIndex -lt 0) {
        throw "Could not locate root injection point in $Path"
    }

    if ($fileTree.Contains('data-path="index.md"')) {
        Set-Content -LiteralPath $Path -Value $fileTree -Encoding UTF8
        return
    }

    $updatedFileTree = $fileTree.Insert($folderIndex, $rootEntries)

    Set-Content -LiteralPath $Path -Value $updatedFileTree -Encoding UTF8
}

foreach ($pageSpec in $pageSpecs) {
    $sourcePath = Join-Path $VaultPath $pageSpec.Source
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Source markdown not found: $sourcePath"
    }

    $markdown = Get-Content -LiteralPath $sourcePath -Raw
    $linkedMarkdown = Convert-WikiLinks -Markdown $markdown -Map $linkMap
    $conversion = Convert-MarkdownBody -Markdown $linkedMarkdown
    $description = Get-MarkdownDescription -Markdown $markdown
    $outlineHtml = New-OutlineHtml -PagePath $pageSpec.Output -Headings $conversion.Headings
    $contentBlock = New-ContentBlock -Title $pageSpec.Title -BodyHtml $conversion.Html
    $pageHtml = Apply-PageTemplate -Template $template -OutputPath $pageSpec.Output -Title $pageSpec.Title -Description $description -ContentBlock $contentBlock -OutlineHtml $outlineHtml
    $outputPath = Join-Path $ExportPath $pageSpec.Output

    Set-Content -LiteralPath $outputPath -Value $pageHtml -Encoding UTF8
    Write-Host "Updated $outputPath"
}

Update-FileTreeRootEntries -Path $fileTreePath
Write-Host "Updated $fileTreePath"
