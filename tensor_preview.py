'''
텐서는 pytorch의 기본 단위이며 GPU 연산을 가능하게 한다. 
또한 numpy의 배열과 유사하여 손쉽게 다룰 수 있다.
'''

#1.1 텐서 만들기 
 # 라이브러리를 불러온다. 

import torch
import numpy as np

# 빈 텐서 생성
x = torch.empty(5,4)
print("x의 값 : {} " .format(x))
print("x의 타입은 : {} 입니다." .format(x.dtype))

ones = torch.ones(3,3)
zeros = torch.zeros(2)
random = torch.rand(5,6)

print("ones tensor data : {}" .format(ones))
print("zeros tensor data : {}" .format(zeros))
print("random tensor data : {}" .format(random))

#1.2 리스트, 넘파이 배열을 텐서로 만들기
l = [13,14]
r = np.array([4,56,7])

torch.tensor(l)
torch.tensor(r)

print("l 데이터는 : {}입니다." .format(l))
print("r 데이터는 : {}입니다." .format(r))

print("l 타입은 : {}입니다." .format(type(torch.tensor(l))))
print("r 타입은 : {}입니다." .format(type(torch.tensor(r))))

print("l 사이즈는 : {}입니다." .format(torch.tensor(l).size()))
print("r 사이즈는 : {}입니다." .format(torch.tensor(r).size()))

#1.3 텐서의 덧셈
x = torch.rand(2,2)
print("x의 값은 {}입니다." .format(x))
y = torch.ones(2,2)
print("y의 값은 {}입니다." .format(y))


print("x+y의 값은 {}입니다." .format(x+y))
print("x+y의 값은 {}입니다." .format(torch.add(x,y)))
print("x+y의 값은 {}입니다." .format(y.add(x)))

#1.4 텐서의 크기 변환하기
x = torch.rand(8,8)
print("x의 크기는 : {}" .format(torch.tensor(x).size()))
 # 2차원을 1차원인 64벡터로 텐서가 변했다는 뜻이다. 
a = x.view(64) 
print("a의 크기는 : {}" .format(a.size()))
 # 3차원으로 변환, -1은 2, 3항의 값을 기준으로 자동으로 매칭 해준다.
b = x.view(-1,4,2)
print("b의 크기는 : {}" .format(b.size()))

#1.5 텐서에서 넘파이로 만들기
x = torch.rand(3,3)
print("y의 타입은 : {} 입니다." .format(type(x)))
y = x.numpy()
print("y의 타입은 : {} 입니다." .format(type(y)))

#1.6 단일 텐서에서 값으로 뽑아내기
 #단일 텐서인 경우에서만 동작한다. 
z = torch.ones(1)
print(z.item())
