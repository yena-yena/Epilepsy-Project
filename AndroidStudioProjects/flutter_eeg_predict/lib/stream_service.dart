import 'dart:convert';
import 'package:http/http.dart' as http;

class StreamService {
  static const String streamUrl = 'http://10.0.2.2:8001/stream';

  // 전치 함수: [channel][time] → [time][channel]
  static List<List<double>> transpose(List<List<double>> matrix) {
    return List.generate(
      matrix[0].length,
          (i) => List.generate(matrix.length, (j) => matrix[j][i]),
    );
  }

  // EEG 데이터 받아오기
  static Future<Map<String, dynamic>?> fetchNextEEG() async {
    try {
      print("🌐 StreamService: $streamUrl 요청 시작");

      final response = await http.get(
        Uri.parse(streamUrl),
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        },
      ).timeout(const Duration(seconds: 5));

      print("📡 StreamService 응답 상태: ${response.statusCode}");

      if (response.statusCode != 200) {
        print('❌ EEG stream 오류: ${response.statusCode} - ${response.body}');
        return null;
      }

      final Map<String, dynamic> json = jsonDecode(response.body);
      print("📊 받은 데이터: ${json}");

      if (json['data'] == null) {
        print("❌ 'data' 필드가 없습니다");
        return null;
      }

      // [channel][time] 데이터를 double로 변환하고 스케일링
      final List<List<double>> original = (json['data'] as List<dynamic>)
          .map<List<double>>((row) {
        return (row as List<dynamic>).map<double>((e) {
          double value;
          if (e is int) {
            value = e.toDouble();
          } else if (e is double) {
            value = e;
          } else if (e is String) {
            value = double.tryParse(e) ?? 0.0;
          } else {
            value = 0.0;
          }
          return value * 10000;
        }).toList();
      }).toList();

      final List<List<double>> transposed = transpose(original);

      print("📊 변환된 데이터 shape: ${transposed.length} x ${transposed[0].length}");

      return {
        'data': transposed,  // ✅ [time][channel]로 반환
        'label': json['label'],
      };
    } catch (e) {
      print('❌ StreamService 연결 오류: $e');
      return null;
    }
  }
}
