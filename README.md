# TTS-Tacotron2

tts by keras

### 선정 배경

> 음성 기술은 크게 3가지 `음성인식`, `음성합성`, `음성변환`으로 나뉜다. 음성인식은 음성을 text로 변환하는 기술이고 합성은 text를 음성으로 변환은 음성을 음성으로 변환하는 기술이다.
>
> 그 중 오디오 북에 관심이 많아서 음성합성 기술인 tts(text to speach)를 선정하였다.

> tts 기술을 구현하기 위해서는 여러가지 기술이 발표 되었는데 그중 대표적인 것이 deepVoice 1,2,3과 tacotron 1,2이다.

1. tacotron2는 참고할 수 있는 자료가 다른 것에 비해 많았다.
2. tacotron2는 가장 최근에 발표된 논문이였다.
3. attention을 활용해서 정확도가 높았다.

### 사용방법

[학습시]

``python train.py 0``  ## 0,1,2,3,4,5

in 폴더에 학습양을 분배하기 위해 transcript_{}.txt로 파일을 나눠두었다. 0은 33개 데이터의 정보만 있는 테스트 데이터이다.

[합성시]

``python test.py 안녕하세요``

안녕하세요 대신 변환하고자 하는 텍스트를 넣으면 된다.

### 작동과정

[encoder]

![encoder](https://user-images.githubusercontent.com/62198488/87370047-ee414580-c5bc-11ea-8a8c-af273a2c43fa.png)

[decoder]

![decoder](https://user-images.githubusercontent.com/62198488/87369975-c0f49780-c5bc-11ea-8f42-5218de3db443.png)

[taco2]

![taco2](https://user-images.githubusercontent.com/62198488/87370084-ff8a5200-c5bc-11ea-9451-df8a73c48e63.png)

### 파일 명명 규칙

### 병렬처리

```
   strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
```

mirroredstrategy를 통해 keras코드 변환을 최소화 하면서 gpu병렬 처리를 사용하였다.

### colab이용시  tensorboard 사용법

[사용]

``%load_ext tensorboard``

``%tensorboard --logdir {log_dir}``

[사용중지]

`!ps aux | grep tensorboard`  # pid 확인

`!kill {PID}`

### 문제점과 해결



### 현재 진행과정(문제점)

![error](https://user-images.githubusercontent.com/62198488/87370647-825fdc80-c5be-11ea-9162-ff0c5b856edc.png)

학습중에 loss가 갑자기 사라진다. 데이터의 값을 확인해보니 모두

nan값으로 유실된다.

### ref

[Ananas120](https://github.com/Ananas120/Tacotron2_in_keras)
