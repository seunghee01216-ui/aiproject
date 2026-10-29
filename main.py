import streamlit as st
st.title('나의 첫 웹 서비스 만들기!')
a=st.text_input('이름을 입력해주세요!')
b=st.selectbox('좋아하는 디저트 선택',['까눌레','마카롱','타르트'])
if st.button('인사말 생성'):
  st.write('반갑습니다!')
  st.info(a+'님 안녕!')
  st.warning(b+'를 좋아하시나바')
  st.error('잘부탁')
  st.balloons()
