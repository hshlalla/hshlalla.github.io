---
categories : 
- linux
title : "리눅스에서 env 설치 사용하기"
tags:
- linux
- pyenv
- vscode
last_modified_at:
---

# 상황
> - 그동안 잘 사용하던 pycharm에서 vscode로 변경하기로 했다. IDE는 웬만하면 바꾸지 않는 경우가 많은데 개발자로 일을 하다보면 python만 사용할수 없어 다른언어에도 대응이 필요할 때가 많다. intellij로 사용해 인터프리터만 python을 사용할까 했다가
vscode가 범용으로 사용하기 가장 유용할것 같아서 vscode로 변경하기로 했다. 또 jupyter도 인앱으로 구동이 가능하니 편리한 부분이 많아보였다. 주로작업환경이 python 3.10인데 전에 3.8버전으로 작업하던 작업들이 있어서 pyenv를 설치하기로 했다.
vscode가 venv는 지원하기 때문에 인터프리터 관리용도로 사용하려고 한다.

# pyenv
> 사전에 필요한 패키지 설치
``` shell
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl \
git
```
pyenv 설치전에 위에 패키지들을 설치해야 구동이 되므로 복사해서 묻따 설치하자


### pyenv git에서 설치
``` shell
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
자신의 홈폴더에서 .pyenv가 생성된다.

### 환경설정
> vi를 이용해서 수정해도 되고 복붙해도 된다.
> 여기까지 진행했으면 설치는 완료된것이니 shell에서 pyenv를 쳐보자

``` shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

### 설치가능한 버전확인
> python 여러버전이 보인다면 성공 추후 python 버전업이 되면 pyenv에 누락되는경우가 있으니 업데이트를 진행한다. 
``` shell
pyenv update
pyenv install --list
```


### pyenv install 하기
> 원하는 버전을 설치하면된다. 간단하게 완료된다.
> 설치된 버전을 확인하려면 아래 명령어를 확인하자.

``` shell
pyenv install 3.7.3
pyenv versions
```

# 추가로 virtualenv는 설치해서 관리하면 편하지만 난 vscode에서 venv를 사용하려고 따로 설치하지는 않았다.
> virtualenv가 필요한경우 설치하자.
```shell
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
```
> pyenv virtualenv 설치
```shell
pyenv virtualenv 3.x.y my_env
```
> 실행,종료
```shell
pyenv activate my_env
pyenv deactivate
```
