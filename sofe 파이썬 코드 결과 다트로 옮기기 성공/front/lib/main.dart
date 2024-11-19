import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chess App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: 'Roboto', // 한글을 지원하는 기본 폰트 사용
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  Future<String> fetchHomeData() async {
    final response = await http.get(Uri.parse('http://172.20.10.2:5000/home'));

    if (response.statusCode == 200) {
      var data = json.decode(response.body);
      return data['message'] ?? 'No message available';
    } else {
      throw Exception('Failed to load home data: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Homepage')),
      body: FutureBuilder<String>(
        future: fetchHomeData(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    snapshot.data ?? 'No data available',
                    style: TextStyle(fontSize: 20),
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => GameResultsPage()),
                      );
                    },
                    child: Text('View Game Results'),
                  ),
                  SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => ReviewPage()),
                      );
                    },
                    child: Text('Write a Review'),
                  ),
                ],
              ),
            );
          }
        },
      ),
    );
  }
}

class GameResultsPage extends StatefulWidget {
  @override
  _GameResultsPageState createState() => _GameResultsPageState();
}

class _GameResultsPageState extends State<GameResultsPage> {
  List<dynamic> results = [];
  String errorMessage = '';

  @override
  void initState() {
    super.initState();
    listenToGameResults();
  }

  void listenToGameResults() async {
    try {
      final request = http.Request('GET', Uri.parse('http://172.20.10.2:5000/game-results'));
      final response = await request.send();

      // 스트림을 통해 SSE 데이터 수신
      response.stream
          .transform(const Utf8Decoder())
          .transform(const LineSplitter())
          .listen((line) {
        if (line.startsWith("data: ")) {
          final jsonData = line.substring(6).trim();
          if (jsonData.isNotEmpty) {
            final data = json.decode(jsonData);

                if (data['status'] == 'complete') {
                    // 완료 상태 처리
                    setState(() {
                        errorMessage = data['message'];
                    });
                } else {
                    // 새로운 데이터 추가
                    setState(() {
                        results.add(data);
                        errorMessage = ''; // 기존 메시지를 초기화
                    });
                }
          }
        }
      }, onError: (error) {
        setState(() {
          errorMessage = '오류 발생: $error';
        });
      });
    } catch (e) {
      setState(() {
        errorMessage = 'SSE 연결 실패: $e';
      });
    }
  }

@override
Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('게임 결과')),
        body: Padding(
            padding: EdgeInsets.all(16.0),
            child: Column(
                children: [
                    if (errorMessage.isNotEmpty)
                        Text(errorMessage, style: TextStyle(color: Colors.green)),
                    Expanded(
                        child: results.isEmpty
                            ? Center(child: CircularProgressIndicator())
                            : ListView.builder(
                                itemCount: results.length,
                                itemBuilder: (context, index) {
                                    var result = results[index];
                                    return Card(
                                        child: ListTile(
                                            title: Text('턴: ${result['turn'] ?? 'N/A'}'),
                                            subtitle: Column(
                                                crossAxisAlignment: CrossAxisAlignment.start,
                                                children: [
                                                    Text('이동: ${result['moves'] ?? 'N/A'}'),
                                                    Text('상태: ${result['board_state'] ?? 'N/A'}'),
                                                ],
                                            ),
                                        ),
                                    );
                                },
                            ),
                    ),
                ],
            ),
        ),
    );
}

}

class ReviewPage extends StatefulWidget {
  @override
  _ReviewPageState createState() => _ReviewPageState();
}

class _ReviewPageState extends State<ReviewPage> {
  TextEditingController _controller = TextEditingController();

  Future<void> submitReview(String reviewText) async {
    try {
      final response = await http.post(
        Uri.parse('http://172.20.10.2:5000/review'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'review_text': reviewText}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: Text('Review Saved'),
            content: Text('Your review: ${data['saved_review']}'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text('OK'),
              ),
            ],
          ),
        );
      } else {
        throw Exception('Failed to save review');
      }
    } catch (e) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('Error'),
          content: Text('An error occurred: $e'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('OK'),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Write a Review')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(labelText: 'Enter your review'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                if (_controller.text.isNotEmpty) {
                  submitReview(_controller.text);
                } else {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('Error'),
                      content: Text('Review cannot be empty'),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: Text('OK'),
                        ),
                      ],
                    ),
                  );
                }
              },
              child: Text('Submit Review'),
            ),
          ],
        ),
      ),
    );
  }
}
