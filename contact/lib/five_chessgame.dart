import 'package:flutter/material.dart';

class FiveChessGamePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('체스 게임 화면'),
      ),
      body: Center(
        child: Text(
          '체스 게임 화면입니다!',
          style: TextStyle(fontSize: 24),
        ),
      ),
    );
  }
}
