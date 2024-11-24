from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import os
from chess_logic import detect_and_crop_chessboard, detect_moves
import json
from pymongo import MongoClient
import time
import uuid

# Flask 및 CORS 초기화
app = Flask(__name__)
CORS(app)

# MongoDB 클라이언트 연결
mongo_client = MongoClient("mongodb://localhost:27017/")  # MongoDB 연결 문자열
db = mongo_client['chess_db']  # 데이터베이스 이름

board_state = []
turn_count = 1
is_move_Castling = []


def initialize_board_state():
    """board_state와 관련 변수를 초기화합니다."""
    global board_state, is_move_Castling, turn_count
    turn_count = 1
    is_move_Castling = [[0, 0, 0], [0, 0, 0]]
    board_state = [
        ["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
        ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],
        ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"]
    ]


def generate_game_results(game_collection_name):
    """SSE로 처리 상태를 실시간으로 스트리밍."""
    global board_state, turn_count

    initialize_board_state()  # 상태 초기화

    for i in range(1, 13):  # 이미지가 총 12개라고 가정
        try:
            # 이미지 경로 생성
            imageA_path = rf'back\uploads\test ({i}).png'
            imageB_path = rf'back\uploads\test ({i+1}).png'

            # 이미지 경로 확인
            if not os.path.exists(imageA_path) or not os.path.exists(imageB_path):
                yield f"data: {json.dumps({'error': f'Image files missing: {imageA_path}, {imageB_path}'}, ensure_ascii=False)}\n\n"
                break

            # 이미지 처리
            imageA = detect_and_crop_chessboard(imageA_path)
            imageB = detect_and_crop_chessboard(imageB_path)

            if imageA is None or imageB is None:
                yield f"data: {json.dumps({'error': f'Failed to process images: {imageA_path}, {imageB_path}'}, ensure_ascii=False)}\n\n"
                break

            # 이동 감지 및 결과 저장
            moves, board_state = detect_moves(imageA, imageB)

            result = {
                "turn": turn_count,
                "moves": moves,
                "board_state": board_state,
            }

            # MongoDB에 데이터 저장
            db[game_collection_name].insert_one({
                "turn": turn_count,
                "moves": moves,
                "board_state": board_state
            })

            turn_count += 1
            # SSE 형식으로 데이터 스트리밍
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            break

    # 스트리밍 종료 메시지
    yield "data: {\"status\": \"complete\", \"message\": \"All processing done!\"}\n\n"

@app.route('/start-game', methods=['POST'])
def start_game():
    """새 게임을 시작하고 해당 게임의 고유 컬렉션 이름을 반환"""
    global board_state, turn_count, is_move_Castling  # 전역 변수 초기화

    # 체스판 상태 초기화
    turn_count = 1
    is_move_Castling = [[0, 0, 0], [0, 0, 0]]
    board_state = [
        ["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
        ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],
        ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"]
    ]
    # 고유한 게임 컬렉션 이름 생성
    timestamp = int(time.time())  # Unix timestamp 사용
    game_collection_name = f"game_{timestamp}_{uuid.uuid4().hex[:6]}"

    # MongoDB에 초기화 정보 저장
    db[game_collection_name].insert_one({"message": "Game initialized", "timestamp": timestamp})

    return jsonify({"game_collection": game_collection_name, "message": "Game started"})


@app.route('/game-results/<game_collection>', methods=['GET'])
def game_results(game_collection):
    """SSE 엔드포인트로 클라이언트에 스트리밍."""
    return Response(generate_game_results(game_collection), content_type='text/event-stream; charset=utf-8')


@app.route('/game-history', methods=['GET'])
def game_history():
    """MongoDB에 저장된 게임 기록 조회"""   
    # 모든 컬렉션의 이름을 가져와 반환
    collections = db.list_collection_names()
    return jsonify({"game_collections": collections})


@app.route('/game-history/<game_collection>', methods=['GET'])
def specific_game_history(game_collection):
    """특정 게임의 기록을 조회"""
    results = list(db[game_collection].find({}, {"_id": 0}))  # `_id` 필드를 제외하고 모든 문서 가져오기
    return jsonify({"game_history": results})

@app.route('/home', methods=['GET'])
def home():
    """기본 홈 엔드포인트"""
    return jsonify({
        "message": "Welcome to the Homepage! This is your app's backend."
    })

@app.route('/review', methods=['POST'])
def review():
    data = request.get_json()
    review_text = data.get('review_text')
    db['reviews'].insert_one({"review_text": review_text})
    return jsonify({"message": "Review saved successfully!"})

@app.route('/update-game-title/<game_collection>', methods=['POST'])
def update_game_title(game_collection):
    """게임 리스트에서 제목을 업데이트"""
    data = request.get_json()
    new_title = data.get('title')

    if not new_title:
        return jsonify({"error": "Title is required"}), 400

    # MongoDB에서 제목 업데이트
    db[game_collection].update_one(
        {"message": "Game initialized"},  # 초기화 문서만 찾음
        {"$set": {"title": new_title}}
    )

    return jsonify({"message": "Title updated successfully", "title": new_title})




if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)
