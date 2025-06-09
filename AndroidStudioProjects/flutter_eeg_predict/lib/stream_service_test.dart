import 'dart:convert';
import 'package:http/http.dart' as http;

class StreamServiceTest {
  static const String apiUrl = 'http://10.0.2.2:8001/stream';

  static Future<Map<String, dynamic>?> fetchNextEEG() async {
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        final Map<String, dynamic> json = jsonDecode(response.body);

        // ✅ 먼저 숫자로 확실히 변환해두기
        final List<List<double>> raw = (json['data'] as List<dynamic>)
            .map<List<double>>((row) {
          return (row as List<dynamic>).map<double>((e) {
            if (e is int) return e.toDouble();
            if (e is double) return e;
            if (e is String) return double.tryParse(e) ?? 0.0;
            return 0.0;
          }).toList();
        })
            .toList();

        // ✅ 전치: [time][channel] → [channel][time]
        final int timeLen = raw.length;
        final int channelLen = raw[0].length;

        final transposed = List.generate(
          channelLen,
              (i) => List.generate(timeLen, (j) => raw[j][i]),
        );

        return {
          'data': transposed,
          'label': json['label'],
        };
      } else {
        print('❌ 서버 오류: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('❌ 연결 오류: $e');
      return null;
    }
  }
}