import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart'; // 권한 요청을 위한 패키지
import 'second_screen.dart';
import 'third_camera.dart'; // ThirdCameraScreen이 정의된 파일
import 'fourth_review.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // 앱 시작 시 권한 요청
    requestPermissions();

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/', // 첫 화면을 초기 경로로 설정
      routes: {
        '/': (context) => MainScreen(), // 첫 화면
        '/second': (context) => SecondScreen(), // SecondScreen 연결
        '/third_camera': (context) => ThirdCameraScreen(), // ThirdCameraScreen 연결
        '/fourth_review': (context) => FourthReviewScreen(), // FourthReviewScreen 연결
      },
    );
  }

  // 권한 요청 함수
  void requestPermissions() async {
    // 카메라 권한 요청
    if (await Permission.camera.request().isDenied) {
      print('카메라 권한이 거부되었습니다.');
    }
    // 저장소 권한 요청
    if (await Permission.storage.request().isDenied) {
      print('저장소 권한이 거부되었습니다.');
    }
  }
}

class MainScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: GestureDetector(
          onTap: () {
            Navigator.pushNamed(context, '/second'); // SecondScreen으로 이동
          },
          child: Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/first_ui.png'), // 첫 화면 이미지
                fit: BoxFit.contain, // 갤럭시 노트 20 비율에 맞게 이미지 표시
              ),
            ),
          ),
        ),
      ),
    );
  }
}
