# CNN 모델 개념

## 기존의 MLP 기반 이미지 처리의 한계

이미지를 일렬로 나열하게 되면 기존에 이미지가 가진 공간적 특성이나 의미를 파악하기 어렵다(공간정보: Spatial information 을 잃게 된다.)

## CNN(Convolution Neural Network)이란? 

<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2Fn51Co%2FbtqSpGbKtig%2FAAAAAAAAAAAAAAAAAAAAAMtAliKzPBDjvIhapgkS-u1qQInhrQCaNBUC5Pq0cNj2%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3DhiJKwsRMRB%252FsgkHAMR5yFVmfsC0%253D'>

전형적인 CNN 구조는 Conv layer를 몇 개 쌓고, 그 다음에는 Pooling layer를 쌓는 구조를 반복한다.
네트워크를 통과하여 진행할수록 이미지는 점점 작아지지만 Conv layer 때문에 일반적으로 깊어지고 더 많은 특성 맵을 가지게 된다. 또한 Fully Connected layer로 구성된 일반적인 신경망이 추가되어 마지막 층에서는 sigmoid나 softmax로 예측을 출력한다. 

### 시각 피질

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FcztSDZ%2FbtqSjB28esx%2FAAAAAAAAAAAAAAAAAAAAAKVDc3gTOUsUXav2HnPl22z_8EEjJTUkM2WH3x7NYcBS%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3DHNCmBQTJle5b3Ouvyz9hAy1nqGA%253D">

데이비드 허블과 토르스텐 비셀은 1958년과 1959년에 시각 피질의 구조에 대한 결정적인 인사이트를 제공한 고양이 실험을 연속해서 수행하였는데, 시각 피질 안의 많은 뉴런이 작은 **국부 수용장(local receptive field)** 을 가진다는 것을 확인하였다.이는 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응한다는 것이다. 또한 시각피질은 고수준 뉴런이 이웃한 저수준 뉴런의 출력에 기반한다는 아이디어로 이어졌다.

### 합성곱 층

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FeDAzrg%2FbtqSGiAgoUS%2FAAAAAAAAAAAAAAAAAAAAAL3tLZaYCvHoSvQ7pd3EtsrJEnLeLoAyQ3BeKZj_BYpL%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3Degsqo4PpTCEoH4pbxWaTdYkKbsc%253D">

첫 번째 합성곱 층(Convolutional layer)의 뉴런은 입력 이미지의 모든 픽셀에 연결되는 것이 아니라 합성곱 층 뉴런의 수용장 안에 있는 픽셀에만 연결된다(Sparse 하다). 두, 세, N번째 합성곱 층은 이전의 층의 작은 사각 영역 안에 뉴런과 연결하면서 계층적 구조를 구성한다. 

- **스트라이드(Stride)**: 한 수용장과 다음 수용장 사이의 수평 또는 수직 방향 스텝 크기로, 스트라이드가 크다면 기존의 출력 중 일부만 뽑는 것과 동일한 효과를 갖는다. 

- **필터(filter)** 또는 **커널(kernel)**: 

<img scr="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FxPx28%2FbtqSmBIyh7G%2FAAAAAAAAAAAAAAAAAAAAAE1nfJpWX_H0gjIv6tBZFI3YLmltlO-Qovc3cXTsy58u%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3DId1QfjU4dg%252BNbYQkW1ZuxWQDuIo%253D">

뉴런의 가중치는 수용장 크기의 작은 이미지로 표현될 수 있는데 이것이 필터이며 서로 다른 특성 맵을 산출한다. 예를 들어, 수직 필터의 경우 이미지에서 수직 특성만 추출하여 특성 맵(feature map)을 반환하고 수평 필터의 경우 이미지에서 수평 특성만 추출하여 특성 맵을 반환한다. 

- **풀링(Pooling)**: 

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FdbH4KA%2FbtqSxB8H7ko%2FAAAAAAAAAAAAAAAAAAAAADTy2QZ0o1nvwVOnzSS1zIGeGmFf6E_P7Jw5jqOABcle%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1753973999%26allow_ip%3D%26allow_referer%3D%26signature%3DZI%252BFKeepf9yigp2hvy%252FLnYp8csQ%253D">

CNN의 문제점 중의 하나는 많은 양의 RAM을 차지한다는 것인데, 특히 학습 시 역전파 알고리즘이 역방향 계산을 수행할 때, 정방향에서 계산했던 모든 중간 값들을 필요로 한다. 따라서 계산량과 메모리, 파라미터 수를 줄이기 위해 입력 이미지의 subsample(축소본)을 만드는 것이다. 가중치가 없으면 계산을 하지 않아 학습의 대상이 아닌 단지 입력을 결합해주는 층이다.
  - **최대 풀링(MaxPooling)**: 풀링 층 값 중 최댓값을 반환
  - **평균 풀링(AveragePooling)**: 풀링 층 값을 모두 더한 후 평균값을 반환


참조: Hands On Machine Learning 2nd