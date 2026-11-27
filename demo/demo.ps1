<#
.SYNOPSIS
  Demo de la API de demanda (P1): peticiones reales y comparación contra lo que pasó.

.DESCRIPTION
  Manda a la API los ejemplos de demo\ejemplos (un día de septiembre de 2026 que el modelo no vio al
  entrenar) y compara cada predicción con los viajes reales. Funciona en Windows PowerShell 5.1 y en
  PowerShell 7.

.EXAMPLE
  .\demo\demo.ps1
  .\demo\demo.ps1 -Url http://localhost:8001 -Pausa      # challenger, con pausas entre pasos
#>
param(
    [string]$Url = "http://localhost:8000",
    [switch]$Pausa
)

$ErrorActionPreference = "Stop"
$Url = $Url.TrimEnd("/")
$Ejemplos = Join-Path $PSScriptRoot "ejemplos"
$Real = @{}
Import-Csv (Get-ChildItem (Join-Path $Ejemplos "real_*.csv") | Select-Object -First 1).FullName -Encoding UTF8 |
    ForEach-Object { $Real[$_.ejemplo] = $_ }

function Paso([string]$Titulo) {
    if ($Pausa) { Read-Host "`n[Enter para continuar]" | Out-Null }
    Write-Host ("`n=== $Titulo " + ("=" * [Math]::Max(0, 70 - $Titulo.Length))) -ForegroundColor Cyan
}

function Llamar([string]$Ruta, [string]$Archivo) {
    # Devuelve codigo HTTP, respuesta y milisegundos. Sin archivo hace GET; con archivo, POST del JSON.
    $params = @{ Uri = "$Url$Ruta"; Method = "Get"; UseBasicParsing = $true }
    if ($Archivo) {
        $json = Get-Content (Join-Path $Ejemplos $Archivo) -Raw -Encoding UTF8
        $params.Method = "Post"
        $params.ContentType = "application/json; charset=utf-8"
        $params.Body = [System.Text.Encoding]::UTF8.GetBytes($json)
    }
    $reloj = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-WebRequest @params
        $codigo = [int]$r.StatusCode
        $cuerpo = $r.Content
    } catch {
        $resp = $_.Exception.Response
        if (-not $resp) { throw }
        $codigo = [int]$resp.StatusCode
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $cuerpo = $_.ErrorDetails.Message
        } else {
            $lector = New-Object System.IO.StreamReader($resp.GetResponseStream())
            $cuerpo = $lector.ReadToEnd()
        }
    }
    $reloj.Stop()
    [pscustomobject]@{ Codigo = $codigo; Datos = ($cuerpo | ConvertFrom-Json); Ms = $reloj.ElapsedMilliseconds }
}

Paso "1. Salud del servicio: GET /health"
$r = Llamar "/health"
Write-Host ("HTTP {0} en {1} ms -> estado={2}, modelo={3}, version={4}, uri={5}" -f `
    $r.Codigo, $r.Ms, $r.Datos.estado, $r.Datos.modelo, $r.Datos.version, $r.Datos.uri)
if ($r.Codigo -ne 200) { throw "La API no está lista. Revisa 'docker compose ps' y los logs del contenedor." }

$casos = @(
    @("2. Hora pico: estación concurrida a las 7:00", "01_hora_pico.json", "01_hora_pico"),
    @("3. Madrugada: la misma estación a las 3:00", "02_madrugada.json", "02_madrugada"),
    @("4. Estación nueva (abierta en 2026), hora pico", "03_estacion_nueva.json", "03_estacion_nueva")
)
foreach ($c in $casos) {
    Paso "$($c[0]): POST /predict"
    $fila = $Real[$c[2]]
    Write-Host ("{0}, {1} a las {2:00}:00" -f $fila.estacion, $fila.fecha, [int]$fila.hour)
    $r = Llamar "/predict" $c[1]
    Write-Host ("HTTP {0} en {1} ms -> predicción {2:N1} viajes | reales: {3}" -f `
        $r.Codigo, $r.Ms, [double]$r.Datos.viajes_estimados, $fila.viajes_reales) -ForegroundColor Green
}

Paso "5. Entrada inválida (hour = 25): la API debe responder 422"
$r = Llamar "/predict" "04_invalido.json"
$msg = if ($r.Datos.detail -is [array]) { $r.Datos.detail[0].msg } else { $r.Datos.detail }
Write-Host ("HTTP {0} en {1} ms -> {2}" -f $r.Codigo, $r.Ms, $msg) -ForegroundColor Yellow

Paso "6. Pronóstico de las 24 horas en una sola petición: POST /predict/lote"
$lote = Get-Content (Join-Path $Ejemplos "05_lote_24h.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$r = Llamar "/predict/lote" "05_lote_24h.json"
$fila0 = $Real["05_lote_24h_h00"]
Write-Host ("HTTP {0} en {1} ms | modelo {2} v{3} | {4}, {5}" -f `
    $r.Codigo, $r.Ms, $r.Datos.modelo, $r.Datos.version, $fila0.estacion, $fila0.fecha)
$tabla = for ($i = 0; $i -lt $lote.registros.Count; $i++) {
    $h = [int]$lote.registros[$i].hour
    [pscustomobject]@{
        Hora     = "{0:00}h" -f $h
        Predicho = [Math]::Round([double]$r.Datos.predicciones[$i].viajes_estimados, 1)
        Real     = [int]$Real[("05_lote_24h_h{0:00}" -f $h)].viajes_reales
    }
}
$tabla | Format-Table -AutoSize | Out-String | Write-Host
$totalP = ($tabla | Measure-Object Predicho -Sum).Sum
$totalR = ($tabla | Measure-Object Real -Sum).Sum
$errAbs = ($tabla | ForEach-Object { [Math]::Abs($_.Predicho - $_.Real) } | Measure-Object -Sum).Sum
Write-Host ("Total del día: predicho {0:N0} viajes | real {1} | WAPE {2:N1}%" -f `
    $totalP, $totalR, (100 * $errAbs / [Math]::Max($totalR, 1))) -ForegroundColor Green
