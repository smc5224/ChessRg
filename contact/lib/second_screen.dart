import 'package:flutter/material.dart';

class SecondScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTapUp: (details) {
          // 터치 위치 가져오기
          final RenderBox box = context.findRenderObject() as RenderBox;
          final Offset localPosition = box.globalToLocal(details.globalPosition);

          // 화면 크기 가져오기
          final double screenWidth = MediaQuery.of(context).size.width;
          final double screenHeight = MediaQuery.of(context).size.height;

          // 화면 경계 확인
          if (localPosition.dx < 0 || localPosition.dx > screenWidth || localPosition.dy < 0 || localPosition.dy > screenHeight) {
            // 화면 밖을 터치했을 경우 아무런 반응 없음
            return;
          }

          // 대각선 '/' 기준 y 값 계산
          final double diagonalY = (screenHeight / screenWidth) * localPosition.dx;

          // 대각선 기준 상단/하단 나누기
          if (localPosition.dy < diagonalY) {
            Navigator.pushNamed(context, '/third_camera'); // ThirdCameraScreen으로 이동
          } else {
            Navigator.pushNamed(context, '/fourth_review'); // FourthReviewScreen으로 이동
          }
        },
        child: Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height,
          decoration: BoxDecoration(
            image: DecorationImage(
              image: AssetImage('assets/second_figma.png'), // 두 번째 화면 이미지
              fit: BoxFit.contain, // 갤럭시 노트 20 비율에 맞게 이미지 표시
            ),
          ),
        ),
      ),
    );
  }
}
