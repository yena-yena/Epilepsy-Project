$baseUrl = "https://physionet.org/files/chbmit/1.0.0"
$saveBase = "C:\Epilepsy_Project"

# chb03부터 chb24까지 반복
for ($i = 3; $i -le 24; $i++) {
    $folderName = "chb" + "{0:D2}" -f $i
    $url = "$baseUrl/$folderName/"
    $savePath = Join-Path $saveBase $folderName

    # 폴더 없으면 새로 생성
    if (-not (Test-Path $savePath)) {
        New-Item -ItemType Directory -Path $savePath | Out-Null
    }

    # HTML 파싱하여 EDF 파일 목록 가져오기
    $fileList = Invoke-WebRequest $url
    $matches = Select-String -InputObject $fileList.Content -Pattern "href=""(chb\d{2}_\d{2}.edf)""" -AllMatches
    $edfFiles = $matches.Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

    # EDF 파일 다운로드
    foreach ($file in $edfFiles) {
        $localFile = Join-Path $savePath $file
        if (-not (Test-Path $localFile)) {
            Write-Host "다운로드 중: $folderName/$file"
            Invoke-WebRequest "$url$file" -OutFile $localFile
        }
        else {
            Write-Host "이미 존재함: $folderName/$file - 스킵"
        }
    }
}