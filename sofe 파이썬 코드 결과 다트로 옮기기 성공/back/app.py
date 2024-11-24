from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import os
from chess_logic import detect_and_crop_chessboard, detect_moves, split_chessboard, compare_cells, perform_castling
import json

app = Flask(__name__)
CORS(app)

board_state = []
turn_count = 1
is_move_Castling = []

    # 체스판 초기화 함수 정의
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

def generate_game_results():
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
                yield f"data: {jsonify({'error': f'Image files missing: {imageA_path}, {imageB_path}'})}\n\n"
                break

            # 이미지 처리
            imageA = detect_and_crop_chessboard(imageA_path)
            imageB = detect_and_crop_chessboard(imageB_path)

            if imageA is None or imageB is None:
                yield f"data: {jsonify({'error': f'Failed to process images: {imageA_path}, {imageB_path}'})}\n\n"
                break

            # 이동 감지 및 결과 저장
            moves, board_state = detect_moves(imageA, imageB)

            result = {
                "turn": turn_count,
                "moves": moves,
                "board_state": board_state,
            }

            turn_count += 1
            # SSE 형식으로 데이터 스트리밍
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            break

    # 스트리밍 종료 메시지
    yield "data: {\"status\": \"complete\", \"message\": \"All processing done!\"}\n\n"


@app.route('/home', methods=['GET'])
def home():
    """
    기본 홈 엔드포인트
    """
    return jsonify({
        "message": "Welcome to the Homepage! This is your app's backend."
    })

@app.route('/game1', methods=['POST'])
def game1():
    """
    두 이미지를 비교하여 이동 감지 결과를 반환합니다.
    """
    data = request.get_json()
    image_paths = data.get('images')

    if not image_paths or len(image_paths) != 2:
        return jsonify({"error": "Two image paths are required"}), 400

    imageA = detect_and_crop_chessboard(image_paths[0])
    imageB = detect_and_crop_chessboard(image_paths[1])

    if imageA is None or imageB is None:
        return jsonify({"error": "Failed to process images"}), 500

    moves = detect_moves(imageA, imageB)
    return jsonify({"moves": moves})


@app.route('/game-results', methods=['GET'])
def game_results():
    """SSE 엔드포인트로 클라이언트에 스트리밍."""
    return Response(generate_game_results(), content_type='text/event-stream; charset=utf-8')

# @app.route('/game', methods=['GET'])
# def game():
#     global board_state, turn_count, is_move_Castling
#     results = []  # 결과를 저장할 리스트

#     try:
#         # board_state 초기화
#         initialize_board_state()

#         for i in range(1, 13):  # 이미지가 총 10개라고 가정하고 9개의 턴을 비교
#             # 이미지 경로 생성
#             imageA_path = rf'back\uploads\test ({i}).png'
#             imageB_path = rf'back\uploads\test ({i+1}).png'

#             # 디버깅 메시지 출력
#             print(f"Processing images: {imageA_path}, {imageB_path}")

#             # 이미지 경로 확인
#             if not os.path.exists(imageA_path) or not os.path.exists(imageB_path):
#                 print(f"Error: One or both image paths do not exist: {imageA_path}, {imageB_path}")
#                 return jsonify({"error": f"Image files missing: {imageA_path}, {imageB_path}"}), 400

#             # 이미지 처리
#             imageA = detect_and_crop_chessboard(imageA_path)
#             imageB = detect_and_crop_chessboard(imageB_path)

#             if imageA is None or imageB is None:
#                 print(f"Error: Failed to process images: {imageA_path}, {imageB_path}")
#                 return jsonify({"error": f"Failed to process images: {imageA_path}, {imageB_path}"}), 500

#             # 이동 감지 및 결과 저장
#             moves, board_state = detect_moves(imageA, imageB)

#             if moves:
#                 for move in moves:
#                     results.append({
#                         "turn": turn_count,
#                         "message": move,
#                         "bd": board_state,
#                     })
#             turn_count += 1
            
#             print(results)
#         return jsonify({"results": results})

#     except Exception as e:
#         # 예외 메시지 출력
#         print(f"Error: {str(e)}")
#         return jsonify({"error": str(e)}), 500


@app.route('/review', methods=['GET', 'POST'])
def review():
    """
    리뷰 저장 및 조회 엔드포인트
    """
    if request.method == 'POST':
        data = request.get_json()
        review_text = data.get('review_text')
        return jsonify({
            "message": "Review saved successfully!",
            "saved_review": review_text
        })
    elif request.method == 'GET':
        return jsonify({"message": "Review endpoint is working!"})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)

