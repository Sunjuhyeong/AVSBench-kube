import torch
import torch.nn as nn

# 입력 feature의 크기
input_channels = 1
input_height = 1
input_width = 1

# 확장하려는 feature의 크기
output_channels = 1
output_height = 64
output_width = 64

# ConvTranspose2d 레이어 정의
conv_transpose = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(output_height, output_width))
for p in conv_transpose.parameters():
    print(p)
# print(conv_transpose.parameters())
# 랜덤 데이터를 생성하여 테스트
# random_data = torch.rand(1, input_channels, input_height, input_width)
random_data = torch.ones((1, input_channels, input_height, input_width))

# ConvTranspose2d를 통해 1x1 feature를 HxW로 확장
output_data = conv_transpose(random_data)

# 결과 확인
print(output_data)  # 예상 출력 크기: (1, output_channels, H, W)