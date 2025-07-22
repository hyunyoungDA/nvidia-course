# TOPA 1급 개념 

출처: https://heung-bae-lee.github.io/2019/12/06/deep_learning_01/

<img src="https://heung-bae-lee.github.io/image/Neuron.png">

## 생물학적 뉴런 
수상돌기(dendrite)라는 마뭇가지 모양의 돌기와 축삭돌기(axon)라는 긴 돌기가 존재하는데, 수상돌기는 다른 뉴런에게서 정보(활동전위; Action Potential 또는 신호; Signal)를 수신한다. 수신된 정보는 세포체(cell body)에 누적합이 되는 구조이며, 여러 뉴런에게서 받은 정보의 합이 뉴런 고유의 임곗값(threshold)을 넘으면 축삭돌기를 통해 다른 뉴런에 신호를 전달한다. 이때 신앱스(synapse)가 다른 수상돌기나 세포체에 연결하여 전달한다.

## 퍼셉트론(Perceptron)
가장 간단한 인공 신경망으로, 1957년 프랑크 로젠블라트가 제안했다. 입력과 출력이 이진 값이 아닌 어떤 숫자이며 각각의 입력 연결은 가중치와 연관되어 있다. 
- 다수의 신호(input)을 입력 받아 하나의 신호(output)을 출력하는 것은 뉴런이 전기 신호를 내보내 정보를 전달하는 것과 유사하다.
- 뉴런의 수상돌기나 축삭돌기처럼 신호를 전달하는 역할을 퍼셉트로에서는 weight가 한다.
  - weight가 클수록 해당 신호의 영향력이 높다.

## 활성화 함수

출처: https://jbluke.tistory.com/548

<img src = "https://blog.kakaocdn.net/dna/begt3x/btsEJajixmk/AAAAAAAAAAAAAAAAAAAAAG6xWS4h5oACwj6tS3wrfmKy-3pP5IWZp-Ga1rWqq8Pq/img.jpg?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1753973999&allow_ip=&allow_referer=&signature=LrTqVD0K11MmWwLpnS%2Bmtr5BCa4%3D">

Layer의 연산이 끝나고 나가기 전에 거치는 함수로, 인공신경망이 비선형성 관계를 학습하게 하는 역할이다.
신경망에서 선형 함수를 사용하면 신경망의 층을 깊게 하는 의미가 없어진다. 
- 만약 f(x) = ax를 활성화 함수로 사용한 3층 신경망이 있다고 가정하면, y(x) = f(f(f(x)))처럼 나타나는데, 이 계산은 사실 y(x) = cx와 동일합니다. 즉 은닉층이 없는 네트워크로 표현이 가능하기 때문에 여러층으로 구성하는 이점을 살릴 수 없다. 그러므로 층을 쌓기 위해서는 비선형 함수인 활성화 함수를 사용해야 하고 활성화 함수로는 비선형 함수를 사용해야 한다.

출처: https://velog.io/@leejaejun/AIFFEL-FD-20-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98%EC%9D%98-%EC%9D%B4%ED%95%B4

<img src="https://images.velog.io/images/leejaejun/post/1e12085b-67cf-4807-9bd7-75a3a4e5dac7/image%20(1).png">

  - **Sigmoid**: Sigmoid는 x의 값이 0일때 중심값이 0.5이며 실수 값을 입력 받아서 0 ~ 1 사이의 값으로 압축한다. 다중 분류에서는 Softmax로 활용된다. 하지만 Sigmoid 함수는 다음과 같은 2가지 단점이 존재한다.
    - 기울기 소멸 문제(Vanishing Gradient Problem): 기울기는 입력이 0일 때 가장 크고 절댓값 x가 클수록 기울기는 0에 수렴하면서 역전파 중에 이전의 기울기와 현재 기울기를 곱하면서 점점 기울기가 사라지게 되는 기울기 소멸 문제(Vanishing Gradient Problem)가 발생한다.

    - 시그모이드 함숫값은 0이 중심(zero-centerd) X: 0.5가 중심이기 때문에 공통 부분의 부호가 양수이면 기울기가 무조건 양수가 되고 음수인 경우에는 기울기 값이 무조건 음수가 되어 최적의 경로까지 지그재그로 이동하게 된다. 
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2F1yPdd%2FbtrueyHzCIm%2FAAAAAAAAAAAAAAAAAAAAAC5Kpt29YZI_uKc9NpFAcxzV2fDLsrqKCubIJRyaPD9P%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3DCSibijUkSp6N66xpAJyj6Hn5vHs%253D">

  - **tanh**: 시그모이드와 비슷하나 실수 값을 입력 받아 -1 ~ 1 사이의 값으로 압축한다. 시그모이드와 마찬가지로 기울기 소실 문제를 가지고 있다.

  - **ReLU(Rectified Linear Unit)**: 가장 많이 사용하는 활성화 함수로서, 입력이 0을 넘으면 그 입력을 그대로 출력하고 0 이하면 0을 출력하는 함수이다. 입력이 크더라도 출력 값의 변화량이 줄어들지 않아 간단하면서도 효과적으로 비선형성을 부여한다.

  - **LeakyReLU**: 렐루 함수의 0으로 수렴하는 문제를 보완한 활성화 함수로, 입력이 0 이하인 값이면 입력 값의 일정 비율을 곱하여 0으로 수렴하지 않도록 조정한다.

## 손실 함수

회귀 모델: MSE(Mean Squared Error), MAE(Mean Absoluted Error), R2
이진 분류: Binary Cross Entropy
다중 분류: Categorical Cross Entropy

## 최적화 함수(Optimizer)

  - 경사하강법(Gradient Descent): 손실함수의 값을 최소화하기 위해 모든 학습 데이터에 대한 손실함수의 기울기를 이용하며, 기울기가 증가하는 방향의 반대 방향으로 이동하며 가중치를 조정한다. 즉, 가중치(weight)의 값을 기울기가 0인 최적점으로 이동시키는 것이다.

    - 경사하강법은 손실함수의 값을 최소화 하기 위해 모든 학습 데이터에 대한 기울기를 계산하므로 한번 학습하는데 많은 컴퓨팅 자원을 요구하게 된다.
  
  - 학습률(Learning rate): 가중치를 조정하는 강도를 조절하는 값으로 너무 클 경우 최적점을 찾지 못하고 불안정해질 수 있고 너무 작을 경우엔 학습이 너무 느리게 진행되거나 정체될 수 있다.

  - SGD(Stochastic Gradient Descent): 일부 학습 데이터(mini-batch)에 대한 손실함수의 기울기를 이용하여 전체 데이터를 활용하는 배치 경사하강법보다 정확도는 떨어지지만 같은 시간 내에 더 많이 학습할 수 있다.

<src img = "https://blog.kakaocdn.net/dna/cBX36i/btrQY19wX32/AAAAAAAAAAAAAAAAAAAAAM1j_Sf4L4pm7mISfxZRpT3EzylW8ifXImgo9pQgaadq/img.jpg?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1753973999&allow_ip=&allow_referer=&signature=6fo7%2B3U2kyot4PKo34XbFbw8YUs%3D">

  - Momentum: SGD 기법에 모멘텀을 추가하여 최적화 과정의 속도를 향상시킨 방식으로, 현재 상태의 경사를 무작정 따라가지 않고 마치 관성을 부여한 것처럼 이전 상태의 변화량을 일정 비율로 같이 적용하는 기법이다. 모멘텀 알고리즘을 보면 기울기가 큰 경우 빠르게 이동하고 기울기가 작은 경우 과거 기울기를 통해 안정적으로 이동한다. 

  - Adagrad: 기존의 SGD 기법은 모든 가중치를 같은 학습률로 학습시켰는데 이러한 경우 변화량이 큰 가중치는 너무 크게 조정하고, 반대로 변화량이 작은 가중치는 너무 작게 조정하는 문제가 발생한다. 이를 해결하기 위해 AdaGrad는 학습률이 많이 변화한 Feature는 학습률을 작게 조정하고, 적게 변화한 Feqture는 학습률을 크게 조정한다. 즉, 과거 기울기를 계속 반영하며 기울기가 큰 방향에서는 너무 많은 업데이트를 하지 않도록 적응적 학습률을 활용한다.

  - RMSProp: AdaGrad 기법은 누적 기울기의 제곱근에 반비례한 크기로 가중치를 조정하기 때문에, 어느 순간 이후로는 누적 기울기의 값이 너무 커지면서 너무 빨리 느려져서 학습이 제대로 되지 않는 문제가 발생한다. 이를 방지하기 위해 누적 기울기를 계산할 때 과거의 정보를 어느 정도 잊고 새로운 기울기 정보를 반영하는 방식을 활용한다. 
    - 감쇠율 P: P가 클수록 과거 데이터의 영향력이 크고, 현재 기울기의 가중치 영향력이 작다. 
  
  - Adam(Adaptive Moment Estimation): RMSProp와 Momentum의 장점을 결합한 형태로, 가장 널리 사용되는 최적화 알고리즘이다. 

  - AdamW: 가중치 감쇠(weight decay)라는 규제 기법을 통합한 Adam으로, 각 훈련 반복에서 모델의 가중치에 감쇠 계수를 곱하여 가중치의 크기를 반복적으로 줄인다. 

## 딥러닝 모델 학습의 문제점

딥러닝 모델은 일반화 성능(모델이 학습하지 않은 새로운 데이터에 대해 잘 예측하는 정도)이 좋아야함

- 과적합(Overfitting): 학습 데이터에 너무 딱 들어맞게 학습된 나머지, 일반화 성능이 오히려 낮아지는 문제.

- 가중치 규제(Regularization): 가중치를 규제하여 모델의 가중치 값의 차이가 벌어지는 것을 방지하여 과적합을 방지한다. 
  - L1 규제(Lasso Regularization): 가중치의 절대값의 합의 일정 비율을 손실 함수에 더하는 규제 기법으로, 작은 가중치들을 0에 수렴시켜 피처 선택에 용이하고 큰 가중치 몇 개만 잔류한다.
  - L2 규제(Ridge Regularization): 가중치의 제곱의 합으 일정 비율을 손실 함수에 더하는 규제 기법으로, L1 규제에 비해 0으로 수렴하는 가중치가 적고, 큰 가중치를 더하게 규제한다. 

- 드롭아웃(Dropout): 학습 중 일부 뉴런을 주기적으로 랜덤하게 비활성화하여 모든 뉴런이 과적합 되는 것을 억제하는 기법으로, 검증 및 테스트 과정에서는 적용하지 않음