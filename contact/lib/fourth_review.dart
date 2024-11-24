import 'package:flutter/material.dart';

class FourthReviewScreen extends StatelessWidget {
  // 비동기 데이터 로드 함수
  Future<List<Map<String, dynamic>>> fetchMatchData() async {
    await Future.delayed(Duration(seconds: 2));
    // 임시 데이터
    return [
      {
        "id": 1,
        "result": "흑 승",
        "turns": 2,
        "details": [
          {
            "turn": 1,
            "board": [
              ["R", "N", "B", "Q", "K", "B", "N", "R"],
              ["P", "P", "P", "P", "P", "P", "P", "P"],
              ["", "", "", "", "", "", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["p", "p", "p", "p", "p", "p", "p", "p"],
              ["r", "n", "b", "q", "k", "b", "n", "r"],
            ],
          },
          {
            "turn": 2,
            "board": [
              ["R", "N", "B", "Q", "K", "B", "N", "R"],
              ["P", "P", "P", "P", "P", "", "P", "P"],
              ["", "", "", "", "", "P", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["", "", "", "", "", "", "", ""],
              ["p", "p", "p", "p", "p", "p", "p", "p"],
              ["r", "n", "b", "q", "k", "b", "n", "r"],
            ],
          },
        ],
      },
      {
        "id": 2,
        "result": "백 승",
        "turns": 27,
        "details": [],
      },
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // 상단 이미지 표시
          Container(
            width: double.infinity,
            height: 150,
            child: Image.asset(
              'assets/fourt_review_design.png',
              fit: BoxFit.cover,
            ),
          ),
          // FutureBuilder로 데이터 처리
          Expanded(
            child: FutureBuilder<List<Map<String, dynamic>>>(
              future: fetchMatchData(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasError) {
                  return Center(child: Text('데이터를 불러오는 데 실패했습니다.'));
                }
                if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return Center(
                    child: Text(
                      '경기가 없습니다',
                      style: TextStyle(fontSize: 20),
                    ),
                  );
                }

                final matchData = snapshot.data!;
                return ListView.builder(
                  itemCount: matchData.length,
                  itemBuilder: (context, index) {
                    final match = matchData[index];
                    final isBlackWin = match["result"] == "흑 승";

                    return GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => MatchDetailScreen(match: match),
                          ),
                        );
                      },
                      child: Card(
                        margin: EdgeInsets.symmetric(
                            vertical: 8.0, horizontal: 16.0),
                        elevation: 3,
                        child: Container(
                          padding: EdgeInsets.all(16.0),
                          color: isBlackWin ? Colors.black : Colors.white,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                '경기 ${match["id"]}',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: isBlackWin
                                      ? Colors.white
                                      : Colors.black,
                                ),
                              ),
                              Text(
                                '결과: ${match["result"]} | 턴: ${match["turns"]}',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: isBlackWin
                                      ? Colors.white
                                      : Colors.black,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class MatchDetailScreen extends StatefulWidget {
  final Map<String, dynamic> match;

  MatchDetailScreen({required this.match});

  @override
  _MatchDetailScreenState createState() => _MatchDetailScreenState();
}

class _MatchDetailScreenState extends State<MatchDetailScreen> {
  int selectedTurn = 1;

  @override
  Widget build(BuildContext context) {
    final matchDetails = widget.match["details"];
    final currentBoard = matchDetails[selectedTurn - 1]["board"] as List<List<String>>;

    return Scaffold(
      appBar: AppBar(
        title: Text('경기 ${widget.match["id"]} 상세 화면'),
      ),
      body: Column(
        children: [
          // 체스판 표시
          Expanded(
            flex: 3,
            child: Container(
              padding: EdgeInsets.all(8.0),
              child: AspectRatio(
                aspectRatio: 1.0,
                child: GridView.builder(
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 8,
                  ),
                  itemCount: 64,
                  itemBuilder: (context, index) {
                    int row = index ~/ 8;
                    int col = index % 8;
                    String piece = currentBoard[row][col];
                    return Container(
                      color: (row + col) % 2 == 0 ? Colors.white : Colors.grey,
                      child: Center(child: Text(piece)),
                    );
                  },
                ),
              ),
            ),
          ),
          // 턴 리스트 표시
          Expanded(
            child: ListView.builder(
              itemCount: matchDetails.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text("턴 ${matchDetails[index]["turn"]}"),
                  onTap: () {
                    setState(() {
                      selectedTurn = matchDetails[index]["turn"];
                    });
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
