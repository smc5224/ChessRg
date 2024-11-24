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
        fontFamily: 'Roboto',
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Future<void> startNewGame(BuildContext context) async {
    try {
      // final response = await http.post(Uri.parse('http://172.20.10.2:5000/start-game'));
      final response = await http.post(Uri.parse('http://172.21.53.104:5000/start-game'));
      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        String gameCollectionName = data['game_collection'];
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => GameResultsPage(gameCollectionName: gameCollectionName),
          ),
        );
      } else {
        throw Exception('Failed to start a new game');
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Error starting game: $e'),
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Homepage')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => startNewGame(context),
              child: Text('Start New Game'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => GameHistoryPage()),
                );
              },
              child: Text('Review Games'),
            ),
          ],
        ),
      ),
    );
  }
}

class GameResultsPage extends StatefulWidget {
  final String gameCollectionName;

  GameResultsPage({required this.gameCollectionName});

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
      // final request = http.Request(
      //   'GET',
      //   Uri.parse('http://172.20.10.2:5000/game-results/${widget.gameCollectionName}'),
      // );
      final request = http.Request(
        'GET',
        Uri.parse('http://172.21.53.104:5000/game-results/${widget.gameCollectionName}'),
      );
      final response = await request.send();

      response.stream
          .transform(const Utf8Decoder())
          .transform(const LineSplitter())
          .listen((line) {
        if (line.startsWith("data: ")) {
          final jsonData = line.substring(6).trim();
          if (jsonData.isNotEmpty) {
            final data = json.decode(jsonData);
            if (data['status'] == 'complete') {
              setState(() {
                errorMessage = data['message'];
              });
            } else {
              setState(() {
                results.add(data);
                errorMessage = '';
              });
            }
          }
        }
      }, onError: (error) {
        setState(() {
          errorMessage = 'Error occurred: $error';
        });
      });
    } catch (e) {
      setState(() {
        errorMessage = 'SSE connection failed: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Game Results')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (errorMessage.isNotEmpty)
              Text(errorMessage, style: TextStyle(color: Colors.red)),
            Expanded(
              child: results.isEmpty
                  ? Center(child: CircularProgressIndicator())
                  : ListView.builder(
                      itemCount: results.length,
                      itemBuilder: (context, index) {
                        var result = results[index];
                        return Card(
                          child: ListTile(
                            title: Text('Turn: ${result['turn'] ?? 'N/A'}'),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('Moves: ${result['moves'] ?? 'N/A'}'),
                                Text('State: ${result['board_state'] ?? 'N/A'}'),
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

class GameHistoryPage extends StatefulWidget {
  @override
  _GameHistoryPageState createState() => _GameHistoryPageState();
}

class _GameHistoryPageState extends State<GameHistoryPage> {
  List<dynamic> gameCollections = [];
  Map<String, String> gameTitles = {}; // 게임 ID와 제목 매핑
  String errorMessage = '';

  Future<void> fetchGameHistory() async {
    try {
      // final response = await http.get(Uri.parse('http://172.20.10.2:5000/game-history'));
      final response = await http.get(Uri.parse('http://172.21.53.104:5000/game-history'));
      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        setState(() {
          gameCollections = data['game_collections'];
        });
        // 각 게임의 초기 제목 가져오기
        for (String gameCollection in data['game_collections']) {
          await fetchGameTitle(gameCollection);
        }
      } else {
        throw Exception('Failed to fetch game history');
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Failed to fetch game history: $e';
      });
    }
  }

  Future<void> fetchGameTitle(String gameCollection) async {
    try {
      // final response = await http.get(
      //   Uri.parse('http://172.20.10.2:5000/game-history/$gameCollection'),
      // );
      final response = await http.get(
        Uri.parse('http://172.21.53.104:5000/game-history/$gameCollection'),
      );
      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        setState(() {
          gameTitles[gameCollection] = data['game_history'][0]['title'] ?? gameCollection;
        });
      } else {
        setState(() {
          gameTitles[gameCollection] = gameCollection; // 기본값으로 설정
        });
      }
    } catch (e) {
      setState(() {
        gameTitles[gameCollection] = gameCollection;
      });
    }
  }

  Future<void> updateGameTitle(String gameCollection, String newTitle) async {
    try {
      // final response = await http.post(
      //   Uri.parse('http://172.20.10.2:5000/update-game-title/$gameCollection'),
      //   headers: {'Content-Type': 'application/json'},
      //   body: json.encode({'title': newTitle}),
      // );
      final response = await http.post(
        Uri.parse('http://172.21.53.104:5000/update-game-title/$gameCollection'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'title': newTitle}),
      );
      if (response.statusCode == 200) {
        setState(() {
          gameTitles[gameCollection] = newTitle;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Title updated successfully!')),
        );
      } else {
        throw Exception('Failed to update title');
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error updating title: $e')),
      );
    }
  }

  @override
  void initState() {
    super.initState();
    fetchGameHistory();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Game History')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: gameCollections.isEmpty
            ? (errorMessage.isNotEmpty
                ? Text(errorMessage, style: TextStyle(color: Colors.red))
                : Center(child: CircularProgressIndicator()))
            : ListView.builder(
                itemCount: gameCollections.length,
                itemBuilder: (context, index) {
                  String gameCollection = gameCollections[index];
                  String title = gameTitles[gameCollection] ?? gameCollection;

                  return Card(
                    child: ListTile(
                      title: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: Text(
                              title,
                              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          IconButton(
                            icon: Icon(Icons.edit),
                            onPressed: () async {
                              String? newTitle = await showDialog<String>(
                                context: context,
                                builder: (context) {
                                  TextEditingController titleController =
                                      TextEditingController(text: title);
                                  return AlertDialog(
                                    title: Text('Edit Title'),
                                    content: TextField(
                                      controller: titleController,
                                      decoration:
                                          InputDecoration(hintText: 'Enter new title'),
                                    ),
                                    actions: [
                                      TextButton(
                                        onPressed: () => Navigator.pop(context, null),
                                        child: Text('Cancel'),
                                      ),
                                      TextButton(
                                        onPressed: () =>
                                            Navigator.pop(context, titleController.text),
                                        child: Text('Save'),
                                      ),
                                    ],
                                  );
                                },
                              );
                              if (newTitle != null && newTitle.isNotEmpty) {
                                await updateGameTitle(gameCollection, newTitle);
                              }
                            },
                          ),
                        ],
                      ),
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) =>
                                SpecificGamePage(gameCollection: gameCollection),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
      ),
    );
  }
}


class SpecificGamePage extends StatelessWidget {
  final String gameCollection;

  SpecificGamePage({required this.gameCollection});

  Future<List<dynamic>> fetchSpecificGameDetails() async {
    // final response = await http.get(Uri.parse('http://172.20.10.2:5000/game-history/$gameCollection'));
    final response = await http.get(Uri.parse('http://172.21.53.104:5000/game-history/$gameCollection'));
    if (response.statusCode == 200) {
      var data = json.decode(response.body);
      return data['game_history'];
    } else {
      throw Exception('Failed to fetch game details');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Game Details')),
      body: FutureBuilder<List<dynamic>>(
        future: fetchSpecificGameDetails(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            var gameDetails = snapshot.data!;
            return ListView.builder(
              itemCount: gameDetails.length,
              itemBuilder: (context, index) {
                var turnDetails = gameDetails[index];
                return Card(
                  child: ListTile(
                    title: Text('Turn: ${turnDetails['turn'] ?? 'N/A'}'),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Moves: ${turnDetails['moves'] ?? 'N/A'}'),
                        Text('State: ${turnDetails['board_state'] ?? 'N/A'}'),
                      ],
                    ),
                  ),
                );
              },
            );
          }
        },
      ),
    );
  }
}
