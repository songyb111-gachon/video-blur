# video-blur

영상과 사진 속 사람과 얼굴을 블러나 모자이크로 가리는 파이썬 스크립트입니다. GPU 없이 CPU만으로 돌아갑니다.

| 스크립트 | 가리는 대상 | 대상을 찾는 방법 |
|---|---|---|
| `blurall.py` | 화면에 나오는 **모든 사람**(전신) 또는 **모든 얼굴** | 프레임마다 얼굴 검출(SCRFD), 전신일 때는 사람 분할(YOLO11-seg)도 함께 |
| `vblur.py` | **지정한** 사람·얼굴·물체만 | 한 프레임에서 대상을 짚어 주면 SAM으로 윤곽을 잡아 추적 |
| `imgblur.py` | **사진**의 모든 얼굴, 모든 사람, 또는 지정한 대상 | 위와 같은 모델을 사진 한 장에 사용 |

`app.py`는 이 셋을 브라우저 화면에서 쓰게 해 주는 앱입니다.

영상을 다루는 두 스크립트는 다음을 지킵니다.

- 원본과 같은 해상도, 프레임 수, 프레임 타이밍(가변 프레임 포함)으로 다시 인코딩합니다.
- 10비트 HDR 영상은 HDR10 정보(색 형식, mastering display, MaxCLL)를 유지합니다.
- 원본에 HDR10+ 메타데이터가 있으면 완성본에 프레임별 값을 그대로 복사합니다.
- 소리는 다시 인코딩하지 않고 그대로 복사합니다.
- 가린 영역 밖은 건드리지 않고, 결과를 확인하는 명령(`check`, `residual`, `sheet`)이 들어 있습니다.

## 앱으로 쓰기

명령어 없이 브라우저 화면에서 쓰는 방법입니다. PC에서 처리하고, 화면은 그 PC의 브라우저에 뜹니다. 안드로이드 폰에서 직접 돌리는 방법은 아래 "안드로이드 폰에서 쓰기"에 있습니다.

### 파일 하나로 시작하기

[Releases](https://github.com/songyb111-gachon/video-blur/releases/latest)에서 파일 하나만 받으면 됩니다.

- 윈도우: `video-blur-windows.bat`을 받아 더블클릭합니다. 내려받은 파일이라 윈도우가 경고를 띄우면 "추가 정보"를 누른 뒤 "실행"을 고릅니다.
- macOS, 리눅스: `video-blur-mac-linux.sh`를 받아 터미널에서 `sh video-blur-mac-linux.sh`를 실행합니다.
- 안드로이드: 아래 "안드로이드 폰에서 쓰기"를 따릅니다.

처음 실행하면 앱을 내려받고 아래 표의 것들을 설치한 뒤 브라우저에 앱을 엽니다. 다음부터는 같은 파일을 다시 실행하면 바로 켜집니다. 시작 파일은 어느 폴더에 두고 실행해도 항상 같은 설치 위치를 쓰므로 다시 설치하지 않습니다. 새 버전의 시작 파일을 실행하면 같은 위치의 앱만 새 버전으로 바뀌고, 받아 둔 모델과 올린 파일은 그대로 남습니다. 앱은 윈도우에서는 `%LocalAppData%\video-blur`, macOS와 리눅스에서는 `~/.local/share/video-blur`에 설치되고, 그 폴더를 지우면 앱과 올린 영상이 함께 지워집니다.

### 저장소를 받아서 시작하기

1. 이 저장소를 내려받아 압축을 풉니다(GitHub의 Code > Download ZIP).
2. 윈도우는 `run.bat`을 더블클릭하고, macOS와 리눅스는 터미널에서 `./run.sh`를 실행합니다.
3. 브라우저에 `http://127.0.0.1:8765`가 열립니다. 영상이나 사진을 올리고, 가릴 방식을 고르고, "블러 시작"을 누릅니다.

실행할 때 없는 것은 스스로 설치합니다. 처음 한 번은 인터넷 연결이 필요하고 몇 분 걸립니다.

| 없는 것 | 설치 방법 | 설치되는 곳 |
|---|---|---|
| Python | 윈도우: winget으로 Python 3.12. macOS: Homebrew. 리눅스: apt, dnf, pacman | PC |
| 파이썬 패키지 | pip. "모든 사람" 방식에 쓰는 ultralytics와 torch가 용량이 큽니다 | 앱 폴더의 `.venv` |
| ffmpeg | 윈도우: winget(Gyan.FFmpeg), winget이 없으면 BtbN 빌드를 내려받음. macOS: Homebrew. 리눅스: apt, dnf, pacman | PC, 또는 앱 폴더의 `tools` |
| 모델 파일 | 해당 방식을 처음 쓸 때 앱이 내려받음 | `~/.cache/vblur` |

리눅스에서 시스템 패키지를 설치할 때는 관리자 암호를 묻습니다. 자동 설치가 안 되면 무엇을 직접 설치해야 하는지 창에 표시됩니다. 직접 설치해 쓰려면 Python 3.9 이상과 ffmpeg 5.1 이상(libx265 포함)을 준비하고 `pip install -r requirements.txt` 뒤에 `python app.py`를 실행하면 됩니다.

화면에서 할 수 있는 일:

- **모든 얼굴 / 모든 사람**: 영상 전체 또는 일부 구간을 자동으로 가립니다. 끝나면 원본과 나란히 비교하고, 얼굴인지 애매한 곳을 하나씩 켜고 꺼서 다시 만들 수 있습니다.
- **골라서**: 구간을 불러온 뒤 화면에서 대상을 박스로 감싸고 점을 찍습니다. "윤곽 확인"을 누르면 가려질 윤곽을 바로 보여 줍니다.
- **사진**: jpg, png, webp, bmp 파일을 같은 세 가지 방식으로 가립니다. 찾은 얼굴을 하나씩 켜고 꺼서 남길 얼굴을 정할 수 있습니다. 결과에는 촬영 정보(EXIF)를 넣지 않습니다(아래 "imgblur.py: 사진" 참고).
- 블러/모자이크, 세기, 화질, HDR10+ 유지 여부를 고를 수 있고, 진행률과 남은 시간이 표시됩니다. 중간에 취소하거나 앱을 꺼도 찾아 둔 부분부터 이어서 합니다.

**휴대폰이나 다른 기기에서 접속하기.** 앱 첫 화면의 "휴대폰이나 다른 기기에서 접속하기"를 열어 "다른 기기 접속 켜기"를 누르면 주소와 PIN이 표시됩니다. 같은 와이파이의 휴대폰 브라우저에서 그 주소를 열고 PIN을 입력하면 영상과 사진을 올리고 결과를 내려받을 수 있습니다. 같은 자리에서 끌 수 있고, 켤 때마다 PIN이 새로 정해집니다. 이 스위치는 앱을 켠 기기의 화면에만 나타납니다.

- 켜지 않으면 앱은 그 기기 안에서만 열립니다(127.0.0.1).
- 처음부터 켠 채로 시작하려면 시작 파일 뒤에 `--lan`을 붙입니다: `video-blur-windows.bat --lan`, `sh video-blur-mac-linux.sh --lan`, `run.bat --lan`, `./run.sh --lan`. 윈도우에서는 시작 파일이 있는 폴더의 주소창에 `cmd`를 입력해 창을 열고 위 명령을 실행합니다. 1.0.0에서는 이 방법만 있습니다.
- 윈도우가 방화벽 허용을 물으면 개인 네트워크에서 허용해야 다른 기기가 접속할 수 있습니다.
- 접속한 기기는 올리기, 가리기, 내려받기를 할 수 있지만, 이 스위치와 "파일 경로로 열기"는 쓸 수 없습니다.

### 안드로이드 폰에서 쓰기

PC 없이 폰에서 직접 돌리는 방법입니다. **아직 실제 폰에서 실행해 보지 못했습니다.** 설치가 중간에 멈추면 화면에 나온 내용을 Issues에 알려 주세요.

1. [Termux](https://termux.dev) 앱을 설치합니다(F-Droid나 Termux의 GitHub에서 받을 수 있습니다).
2. Termux를 열고 아래 한 줄을 붙여 넣습니다.

```sh
curl -L -o vb.sh https://github.com/songyb111-gachon/video-blur/releases/latest/download/video-blur-android-termux.sh && sh vb.sh
```

3. 설치가 끝나면 폰의 브라우저에 앱이 열립니다. 다음부터는 Termux에서 `sh vb.sh`만 실행하면 됩니다.

릴리스가 올라오기 전이거나 최신 코드를 써 보려면 아래 줄을 씁니다. 이때는 켤 때마다 `main` 브랜치의 파일을 새로 받습니다.

```sh
curl -L -o vb.sh https://raw.githubusercontent.com/songyb111-gachon/video-blur/main/release/video-blur-android-termux.sh && VB_VER=main sh vb.sh
```

`curl`이 없다고 나오면 `pkg install curl`을 먼저 실행합니다.

알아 둘 점:

- Python, ffmpeg, numpy, Pillow, OpenCV, onnxruntime을 Termux 패키지로 설치합니다. OpenCV가 기본 저장소에 없으면 Termux의 `x11-repo`를 추가합니다. onnxruntime이 기본 저장소에 없으면 커뮤니티 저장소 TUR을 추가할지 물어봅니다.
- "모든 사람" 방식은 폰에서 쓸 수 없습니다(torch가 필요합니다). "모든 얼굴"과 "골라서"만 됩니다.
- 영상은 PC보다 훨씬 오래 걸립니다. 사진이나 짧은 구간부터 해 보세요. 처리하는 동안 Termux를 닫지 마세요.
- 앱은 Termux의 `~/video-blur`에 설치됩니다. 지우려면 Termux에서 `rm -rf ~/video-blur ~/.cache/vblur`를 실행합니다.
- 아이폰에는 이런 방법이 없습니다. PC에서 `--lan`으로 켜고 아이폰 브라우저로 접속해 쓰세요.

알아 둘 점:

- 올린 영상과 작업 파일, 결과는 앱 폴더 안의 `jobs`에 쌓입니다. 화면의 "앱에서 지우기"로 지웁니다.
- 윈도우에서는 경로에 한글이 있으면 OpenCV가 파일을 읽지 못할 수 있습니다. 앱은 이 경우 작업 폴더를 영문 경로로 바꿔 쓰고, 그 위치를 시작할 때 창에 표시합니다. 명령어로 직접 쓸 때는 작업 폴더를 영문 경로에 두세요.
- 메모리는 8GB 이상을 권합니다. "모든 사람" 방식이 가장 많이 씁니다.
- 앱과 자동 설치는 리눅스에서 시험했습니다. 윈도우(`run.bat`, winget 설치), macOS, 안드로이드(Termux)에서는 아직 실행해 보지 못했습니다.
- 옵션: `--port 번호`, `--data 폴더`(작업 폴더 위치), `--no-browser`.

아래는 같은 기능을 명령어로 직접 쓰는 방법입니다.

## 먼저 알아 둘 점

- **빠짐없이 가려진다는 보장은 없습니다.** 멀리 있거나 반쯤 가려진 사람, 아주 작은 얼굴은 놓칠 수 있습니다. 공개할 영상이라면 아래의 확인 단계를 건너뛰지 말고, 완성본을 직접 재생해 보세요.
- 블러는 대상을 흐리게 할 뿐 지우지 않습니다. 형체와 옷 색은 남습니다.
- 장식, 조명, 포스터, 옷에 그려진 캐릭터가 얼굴로 잡혀 흐려질 수 있습니다.
- Dolby Vision 메타데이터는 유지되지 않습니다.

## 설치

필요한 것: Python 3, ffmpeg(libx264와 libx265 포함). HDR 영상을 다룰 때는 ffmpeg에 zscale 필터가 있어야 검출용 화면이 제대로 밝아집니다. 없으면 스크립트가 알려 주고 그대로 진행합니다.

```bash
pip install opencv-python-headless numpy onnxruntime
pip install pillow               # imgblur.py(사진)에 필요
pip install ultralytics          # blurall.py로 전신을 가릴 때만 필요 (torch가 함께 설치됨)
```

모델 파일은 저장소에 들어 있지 않습니다. `~/.cache/vblur`에 내려받습니다(다른 폴더를 쓰려면 환경 변수 `VBLUR_MODELS`).

```bash
mkdir -p ~/.cache/vblur && cd ~/.cache/vblur

# 얼굴 검출 (blurall.py)
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -j -o buffalo_l.zip det_10g.onnx && rm buffalo_l.zip

# 사람 분할 (blurall.py, 전신을 가릴 때만)
curl -L -o yolo11s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt

# SAM (vblur.py)
for f in encoder decoder; do
  curl -L -o vit_b-$f-quant.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-$f-quant.onnx
done
```

## blurall.py: 모든 사람 또는 모든 얼굴

작업 폴더 `W`는 원본과 다른 빈 폴더를 씁니다. 프레임 번호는 0부터입니다.

```bash
B="python3 blurall.py"; W=work

$B probe input.mp4 $W                    # 해상도, 회전, HDR/HDR10+ 여부, 프레임 수
$B overview $W $W/ov.jpg                 # 전체를 24장으로 훑어보기. 검출용 밝기 보정값이 저장됨
$B detect $W --people                    # 얼굴만 가릴 거면 --people 을 뺀다 (약 3배 빠름)
$B link $W                               # 프레임별 얼굴 검출을 시간축으로 잇기
$B suspects $W $W/sus.jpg                # 얼굴인지 애매한 후보를 타일로 보기
$B masks $W --mode people                # 얼굴만: --mode faces
$B sheet $W $W/m.jpg --frames 0:300:30 --alpha --boxes   # 가려질 경계(빨간 선) 미리 보기
$B render $W output.mp4                  # --style mosaic, --strength px, --crf 16, --preset medium
$B check $W output.mp4                   # 숫자 확인
$B residual $W output.mp4 $W/res.jpg     # 완성본에 얼굴 검출을 다시 돌려 남은 얼굴 찾기
$B sheet $W $W/cmp.jpg --frames 0:300:30 --pair output.mp4   # 원본과 완성본을 나란히 보기
```

단계별 설명:

- **detect**: 시간이 가장 오래 걸립니다. 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 합니다. `--a 첫프레임 --b 끝프레임`으로 구간을 주면 그 밖의 프레임은 건드리지 않습니다. 어두운 영상에서 사람을 놓치면 `--gamma 2.0`처럼 더 밝게 줍니다.
- **link**: 한두 프레임 놓친 얼굴을 앞뒤에서 메우고, 처음 잡히기 전과 마지막으로 잡힌 뒤 8프레임도 덮습니다. "suspect" 목록을 출력합니다.
- **suspects**: suspect는 크기는 큰데 점수가 낮은 후보로, 조명이나 장식이 섞여 있습니다. `--people`로 검출했다면 사람 분할 결과와 겹치지 않는 것은 자동으로 빠지고 타일에 `AUTO`가 붙습니다. 타일을 보고 `masks`에 `--drop 3,17`(가리지 않을 id)이나 `--keep 5`(자동으로 빠졌지만 가릴 id)를 줍니다. 애매하면 가리는 쪽이 안전합니다.
- **masks**: `people`은 사람 윤곽에 여유를 두고 앞뒤 2프레임의 결과를 합칩니다. `faces`는 얼굴보다 조금 넓은 부드러운 타원입니다. `--people`로 검출한 작업 폴더에서는 두 모드를 모두 만들 수 있습니다.
- **sheet**: `--frames`는 `a:b:간격` 또는 `12,40,96`. `--crop x0,y0,x1,y1`로 확대합니다. 노란 상자는 검출된 얼굴, 파란 상자는 앞뒤에서 메운 얼굴입니다.
- **check**: 프레임 수와 타이밍, 가린 영역 안이 정말 흐려졌는지(0에 가까울수록 좋음), 그 밖이 원본 그대로인지, HDR10+ 값이 원본과 같은지를 출력합니다. 가려야 할 사람이 빠졌는지는 알려 주지 않습니다.
- **residual**: 타일의 위쪽이 원본, 아래쪽이 완성본입니다. 아래쪽에 실제 얼굴이 또렷이 보이면 놓친 것입니다. 얼굴이 보이는 사람만 잡아 주므로, 전신을 가릴 때 뒤돌아 있는 사람이 빠지지 않았는지는 마지막 `sheet --pair`로 봅니다.

걸리는 시간(CPU 2코어에서 잰 값): 검출은 얼굴만이면 프레임당 약 1초, 전신이면 약 3초입니다. 인코딩은 1080p가 초당 3프레임 안팎, 4K는 `--preset fast`로 초당 1~2프레임입니다.

## vblur.py: 지정한 대상만

좌표는 모두 회전이 적용된 화면 기준 픽셀이고, 프레임 번호는 0부터입니다. 시트 이미지는 축소되어 있으니 좌표는 `--grid` 눈금에서 읽습니다.

```bash
V="python3 vblur.py"; W=work

$V probe input.mp4 $W
$V overview $W $W/ov.jpg                       # 대상이 나오는 구간 찾기. 권장 gamma도 출력
$V frames $W 150 390                           # 그 구간의 미리보기 프레임 뽑기
$V sheet $W $W/s.jpg --frames 150:390:8 --grid # 눈금이 그려진 시트에서 좌표와 프레임 번호 읽기
# $W/job.json 작성 (아래 형식)
$V embed $W                                    # 가장 오래 걸림. 중단돼도 이어서 계산
$V track $W --seed-only                        # 씨앗 윤곽만 먼저 만들어 확인
$V sheet $W $W/seed.jpg --frames 270 --seeds --tracks
$V track $W                                    # 전체 추적
$V sheet $W $W/tr.jpg --frames 150:390:6 --tracks
$V masks $W
$V sheet $W $W/al.jpg --frames 150:390:6 --tracks --alpha   # 빨간 선이 실제로 가려질 경계
$V render $W output.mp4                        # --style mosaic, --strength px, --crf 16, --preset medium
$V check $W output.mp4
$V frames $W 150 390 --src output.mp4 --dir outp            # 완성본을 눈으로 확인
$V sheet $W $W/out.jpg --frames 150:390:6 --dir outp
```

`job.json` 예시:

```json
{
  "range": [154, 386],
  "step": 2,
  "crop": [0, 1040, 1080, 1920],
  "gamma": 2.7,
  "targets": [
    {"name": "A", "ref": 270, "box": [70, 1330, 600, 1920], "pts": [[440, 1480], [300, 1780]]},
    {"name": "B-early", "ref": 198, "box": [655, 1285, 965, 1920], "pts": [[730, 1750], [890, 1420]], "hi": 240},
    {"name": "B-late", "ref": 280, "box": [60, 1370, 340, 1700], "pts": [[220, 1500]], "lo": 262},
    {"name": "sign", "shape": "rect", "keys": {"154": [300, 40, 560, 90], "386": [320, 44, 580, 94]}}
  ]
}
```

- `range`: 처리할 프레임 [첫, 끝]. 대상이 보이기 조금 전부터 사라진 조금 뒤까지 넉넉히 잡습니다.
- `step`: 몇 프레임마다 윤곽을 다시 잡을지. 기본 2. 느린 장면은 3~5, 작고 빠른 대상은 1.
- `crop`(선택): SAM이 볼 영역. SAM은 긴 변을 1024로 줄여서 보므로, 대상이 화면 일부에만 있으면 그 영역만 잡아야 윤곽이 정확합니다.
- `gamma`(선택): 미리보기 밝기. 어두운 영상은 높여야 윤곽이 잘 잡힙니다. gamma나 crop을 바꾸면 embed를 처음부터 다시 계산합니다.
- 대상 항목: `ref`는 대상이 가장 잘 보이는 샘플 프레임(range 시작 + step의 배수), `box`는 그 프레임에서 대상을 감싼 박스, `pts`는 대상 위의 점 1~3개입니다. `lab`을 주면 점마다 1은 대상, 0은 대상이 아님을 뜻합니다. `lo`/`hi`는 이 항목이 맡을 프레임 범위입니다.
- `keys` 항목은 SAM 없이 손으로 놓는 영역입니다. 키 프레임의 박스 사이를 선형 보간하고, 모양은 타원(기본) 또는 `"shape": "rect"`입니다.

잘 안 될 때:

- 윤곽이 옆 사람까지 번지면 box를 좁히고, 번진 곳에 `lab` 0인 점을 추가합니다.
- 가려졌다 다시 나오는 대상은 그 구간에서 `ref`를 새로 잡은 항목을 추가하고 `lo`/`hi`로 구간을 나눕니다(위 예시의 B-early, B-late). 겹쳐도 합쳐집니다.
- 작거나 빠른 대상은 `step`을 줄이고, 그래도 안 되면 `keys` 항목을 덧댑니다.
- `check`의 숫자는 가린 영역이 제대로 흐려졌는지만 알려 줍니다. 가린 위치가 맞는지는 시트를 직접 봐야 합니다.

`embed`는 CPU 4코어 기준으로 샘플 프레임 하나에 6~9초 걸립니다(해상도와 무관).

## imgblur.py: 사진

```bash
python imgblur.py blur 사진.jpg 결과.jpg --mode faces                  # 모든 얼굴
python imgblur.py blur 사진.jpg 결과.jpg --mode people --style mosaic  # 모든 사람, 모자이크
```

- jpg, png, webp, bmp를 읽습니다. 결과 형식은 결과 파일의 확장자를 따릅니다(앱에서는 원본과 같은 형식, bmp는 png). HEIC는 읽지 못합니다. 16비트 png는 8비트로 저장됩니다.
- 크기, 방향(EXIF 회전을 적용해 바로 세움), 색 프로파일(ICC), 투명도(png, webp)는 유지합니다.
- **그 밖의 메타데이터는 결과에 넣지 않습니다.** 위치와 촬영 시간, 기기 정보가 빠지고, 사진 안에 들어 있는 작은 미리보기 그림도 빠집니다. 미리보기에는 가리기 전 모습이 남아 있기 때문입니다. 같은 이유로 모션 포토의 영상 부분이나 HDR 사진의 보조 그림(gain map)처럼 파일에 덧붙은 것도 옮기지 않습니다.
- 큰 사진은 원본에 가까운 크기로 한 번 더 살펴 작은 얼굴을 찾습니다. 1천6백만 화소 사진에서 메모리를 2GB쯤 썼습니다. 1억 2천만 화소가 넘는 사진은 받지 않습니다.
- 지정한 대상만 가리는 것은 앱의 "골라서"에서 합니다(명령어로는 `info`, `seed`, `render --mode pick`을 차례로 쓰는데, 앱이 하는 일을 그대로 따라 하는 것이라 따로 설명하지 않습니다).
- png로 저장하면 가리지 않은 곳이 원본과 똑같습니다. jpg(품질 95)와 webp(품질 95, 손실 압축)는 다시 압축하므로 가리지 않은 곳도 아주 조금 달라집니다(시험한 사진에서 PSNR 43~51 dB). 명령은 저장한 파일을 다시 읽어 이 차이를 `exact`, `psnr`로 출력합니다.

## HDR10+를 넣는 방식

배포판 ffmpeg에 들어 있는 x265는 HDR10+ 지원 없이 빌드된 경우가 많아서, 인코딩할 때 HDR10+를 쓰지 못합니다. 그래서 인코딩이 끝난 뒤 원본 비트스트림에서 프레임별 HDR10+ 메시지(SEI)를 꺼내 완성본 MP4의 같은 프레임에 끼워 넣고, 샘플 크기와 오프셋 표를 고칩니다. 영상을 다시 인코딩하지 않으므로 화면, 타이밍, 소리는 바뀌지 않습니다.

- 들어가는 값은 원본 값 그대로입니다. 블러 전 화면을 기준으로 계산된 값이라, 넓은 영역을 가린 영상에서는 실제 화면과 조금 어긋날 수 있습니다.
- 넣을 수 없는 경우(프레임 수가 다름, 타임스탬프 중복, 분할된 MP4 등)에는 이유를 출력하고 HDR10까지만 유지한 파일을 그대로 둡니다.
- `render`에 `--no-hdr10plus`를 주면 이 단계를 건너뜁니다.

## 시험한 범위

- 1080p 세로 HDR10+ 영상(HEVC 10비트, 가변 프레임 간격)
- 회전 정보가 있는 4K HDR10+ 영상
- 8비트 SDR 영상(H.264)

모두 GPU 없는 리눅스 환경(ffmpeg 6.1, Python 3.13)에서 MP4 파일로 돌렸습니다. 그 밖의 코덱이나 컨테이너에서는 시험하지 않았습니다.

사진은 같은 환경에서 다음으로 시험했습니다: 회전 정보와 색 프로파일이 있는 jpg, 투명도가 있는 png, 16비트 png, 흑백 jpg, webp, bmp, 그림이 두 장 든 jpg(MPO), 6천4백만 화소 jpg. 실제 모션 포토나 Ultra HDR 사진 파일로는 시험하지 못했습니다.

윈도우, macOS, 안드로이드(Termux)에서는 아직 실행해 보지 못했습니다.

## 모델과 라이선스

모델은 이 저장소에 포함되어 있지 않고, 각 모델의 라이선스를 따릅니다.

- 얼굴 검출 SCRFD-10G (`det_10g.onnx`, InsightFace buffalo_l): 비상업 연구용으로만 제공됩니다.
- 사람 분할 YOLO11s-seg (Ultralytics): AGPL-3.0.
- SAM ViT-B (Meta, Segment Anything): Apache-2.0. 여기서 쓰는 ONNX 변환본은 rembg 릴리스에 올라와 있는 파일입니다.

이 저장소의 코드에는 아직 라이선스를 정하지 않았습니다.
