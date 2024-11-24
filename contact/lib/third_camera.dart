import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'five_chessgame.dart'; // five_chessgame.dart 파일을 import

class ThirdCameraScreen extends StatefulWidget {
  @override
  _ThirdCameraScreenState createState() => _ThirdCameraScreenState();
}

class _ThirdCameraScreenState extends State<ThirdCameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  // 직사각형 틀 위치와 크기
  double rectLeft = 0;
  double rectTop = 0;
  double rectWidth = 0;
  double rectHeight = 0;

  // 정사각형 틀 위치와 크기
  double squareLeft = 0;
  double squareTop = 0;
  double squareSize = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final size = MediaQuery.of(context).size;
      setState(() {
        // 직사각형 초기 크기 및 위치
        rectWidth = size.width * 0.2; // 화면 너비의 20%
        rectHeight = size.height * 0.15; // 화면 높이의 15%
        rectLeft = (size.width - rectWidth) / 2; // 화면 중앙 정렬
        rectTop = size.height * 0.5; // 화면 하단 중앙에 배치

        // 정사각형 초기 크기 및 위치
        squareSize = size.width * 0.3; // 화면 너비의 30%
        squareLeft = (size.width - squareSize) / 2; // 화면 중앙 정렬
        squareTop = size.height * 0.2; // 화면 상단 중앙에 배치
      });
    });
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      final firstCamera = cameras.first;

      _controller = CameraController(firstCamera, ResolutionPreset.high);
      _initializeControllerFuture = _controller.initialize();
      setState(() {});
    } catch (e) {
      print('카메라 초기화 중 오류 발생: $e');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('카메라 가이드라인'),
        backgroundColor: Colors.black,
      ),
      body: Stack(
        children: [
          // 카메라 화면
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return CameraPreview(_controller); // 카메라 화면 표시
              } else {
                return Center(child: CircularProgressIndicator()); // 로딩 표시
              }
            },
          ),
          // 정사각형 틀
          Positioned(
            left: squareLeft,
            top: squareTop,
            child: GestureDetector(
              onPanUpdate: (details) {
                setState(() {
                  // 정사각형 틀 이동
                  squareLeft += details.delta.dx;
                  squareTop += details.delta.dy;
                });
              },
              child: Container(
                width: squareSize,
                height: squareSize,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.blue, // 정사각형 틀 색상
                    width: 5, // 테두리 두께
                  ),
                ),
                child: Align(
                  alignment: Alignment.bottomRight, // 크기 조절 핸들 위치
                  child: GestureDetector(
                    onPanUpdate: (details) {
                      setState(() {
                        // 정사각형 크기 조절
                        squareSize += details.delta.dx;
                        if (squareSize < 50) squareSize = 50; // 최소 크기 제한
                      });
                    },
                    child: Icon(Icons.crop_square, size: 20, color: Colors.blue),
                  ),
                ),
              ),
            ),
          ),
          // 정사각형 문구
          Positioned(
            left: squareLeft,
            top: squareTop + squareSize + 10,
            child: Container(
              width: squareSize,
              child: Text(
                '체스판을 인식 시켜 주세요!',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          // 직사각형 틀
          Positioned(
            left: rectLeft,
            top: rectTop,
            child: GestureDetector(
              onPanUpdate: (details) {
                setState(() {
                  // 직사각형 틀 이동
                  rectLeft += details.delta.dx;
                  rectTop += details.delta.dy;
                });
              },
              child: Container(
                width: rectWidth,
                height: rectHeight,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.red, // 직사각형 틀 색상
                    width: 5, // 테두리 두께
                  ),
                ),
                child: Align(
                  alignment: Alignment.bottomRight, // 크기 조절 핸들 위치
                  child: GestureDetector(
                    onPanUpdate: (details) {
                      setState(() {
                        // 직사각형 크기 조절
                        rectWidth += details.delta.dx;
                        rectHeight += details.delta.dy;

                        if (rectWidth < 50) rectWidth = 50; // 최소 너비 제한
                        if (rectHeight < 50) rectHeight = 50; // 최소 높이 제한
                      });
                    },
                    child: Icon(Icons.crop_square, size: 20, color: Colors.red),
                  ),
                ),
              ),
            ),
          ),
          // 직사각형 문구
          Positioned(
            left: rectLeft,
            top: rectTop + rectHeight + 10,
            child: Container(
              width: rectWidth,
              child: Text(
                '턴 넘기기 객체를 인식 시켜 주세요!',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          // 화면 전환 터치 감지 위젯
          Positioned(
            right: 20,
            bottom: 20,
            child: GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => FiveChessGamePage()),
                );
              },
              child: Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.green,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  '체스 게임으로 이동',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
