영상과 사진 속 사람과 얼굴을 블러나 모자이크로 가리는 앱입니다. 아래 파일 하나만 받으면 됩니다.

## 1.1.0에서 달라진 것

- 사진(jpg, png, webp, bmp)도 가릴 수 있습니다. 찾은 얼굴을 하나씩 켜고 꺼서 남길 얼굴을 정할 수 있습니다.
- 휴대폰 같은 다른 기기의 접속을 앱 첫 화면에서 켜고 끌 수 있습니다. `--lan`을 붙여 시작하지 않아도 됩니다.
- 안드로이드용 시작 파일이 생겼습니다(Termux에서 실행).
- 1.0.0을 쓰고 있었다면 아래의 새 시작 파일을 받아 실행하면 됩니다. 같은 위치의 앱만 바뀌고, 받아 둔 모델과 올린 영상은 그대로 남습니다.

## 시작하기

- **윈도우**: `video-blur-windows.bat`을 받아 더블클릭합니다. 내려받은 파일이라 윈도우가 경고를 띄우면 "추가 정보"를 누른 뒤 "실행"을 고릅니다.
- **macOS, 리눅스**: `video-blur-mac-linux.sh`를 받아 터미널에서 `sh video-blur-mac-linux.sh`를 실행합니다.
- **안드로이드**: [Termux](https://termux.dev) 앱을 설치하고, Termux에서 아래 한 줄을 실행합니다. 폰에서는 "모든 사람" 방식을 쓸 수 없고, 영상은 PC보다 훨씬 오래 걸립니다.

```sh
curl -L -o vb.sh https://github.com/songyb111-gachon/video-blur/releases/latest/download/video-blur-android-termux.sh && sh vb.sh
```

처음 실행하면 없는 것을 스스로 설치한 뒤 브라우저에 앱을 엽니다. 인터넷 연결이 필요하고 몇 분 걸립니다. 다음부터는 같은 파일을 다시 실행하면 바로 켜집니다.

## 설치되는 것

| 무엇 | 어디에 |
|---|---|
| 앱 | 윈도우 `%LocalAppData%\video-blur`, macOS·리눅스 `~/.local/share/video-blur`, 안드로이드는 Termux의 `~/video-blur` |
| Python (없을 때) | 윈도우는 winget, macOS는 Homebrew, 리눅스는 시스템 패키지 관리자로 PC에 설치. 안드로이드는 Termux 패키지 |
| 파이썬 패키지 | 앱 폴더의 `.venv`. 안드로이드는 Termux 패키지(필요하면 `x11-repo`를 추가하고, 커뮤니티 저장소 TUR은 물어본 뒤 추가) |
| ffmpeg (없을 때) | 윈도우는 winget, 없으면 앱 폴더의 `tools`에 내려받음. macOS는 Homebrew, 리눅스는 시스템 패키지 관리자 |
| 모델 파일 | 해당 방식을 처음 쓸 때 사용자 폴더의 `.cache/vblur` |

지우려면 앱 폴더를 지웁니다. 올린 영상과 사진, 결과도 그 안에 있습니다.

## 알아 둘 점

- 빠짐없이 가려진다는 보장은 없습니다. 공개하기 전에 결과를 직접 열어 확인하세요.
- 사진 결과에는 촬영 정보(EXIF: 위치, 시간, 기기)와 사진 속 미리보기 그림을 넣지 않습니다.
- 메모리는 8GB 이상을 권합니다.
- 앱과 macOS·리눅스용 시작 파일은 리눅스에서 시험했습니다. 윈도우용과 안드로이드용 시작 파일은 아직 해당 기기에서 실행해 보지 못했습니다.
- 사용법과 모델 라이선스는 [README](https://github.com/songyb111-gachon/video-blur#readme)에 있습니다.
