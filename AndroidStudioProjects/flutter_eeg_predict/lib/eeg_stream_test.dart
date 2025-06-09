import 'dart:async';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'stream_service_test.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class EEGStreamTestGraph extends StatefulWidget {
  final Function(String)? onPrediction;

  const EEGStreamTestGraph({super.key, this.onPrediction});

  @override
  State<EEGStreamTestGraph> createState() => _EEGStreamTestGraphState();
}

class _EEGStreamTestGraphState extends State<EEGStreamTestGraph> {
  List<double> _data = List.filled(50, 0);
  Timer? _timer;
  bool _isStreaming = false;
  List<List<List<double>>> _buffer = [];
  final int _windowSize = 10;

  void _toggleStreaming() {
    if (_isStreaming) {
      _timer?.cancel();
      setState(() => _isStreaming = false);
    } else {
      _startStreaming();
      setState(() => _isStreaming = true);
    }
  }

  void _startStreaming() {
    _timer = Timer.periodic(Duration(seconds: 1), (_) async {
      final eegInfo = await StreamServiceTest.fetchNextEEG();
      if (eegInfo != null) {
        final chunk = eegInfo['data'];
        _buffer.add(chunk);
        if (_buffer.length > _windowSize) {
          _buffer = _buffer.sublist(_buffer.length - _windowSize);
        }
        if (_buffer.length >= _windowSize) {
          final merged = List.generate(
              chunk.length, (ch) => List.generate(_windowSize, (t) => _buffer[t][ch][0]));

          final result = await _sendPredictionRequest(merged);
          if (widget.onPrediction != null) {
            widget.onPrediction!(result);
          }
          setState(() => _data = merged[0]);
        }
      }
    });
  }

  Future<String> _sendPredictionRequest(List<List<double>> data) async {
    const url = 'http://10.0.2.2:8000/predict';
    try {
      final payload = {"data": data};
      final response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        print("🎯 응답 JSON: $result"); // 추가!
        final prediction = result['prediction'] ?? 'Unknown';
        final probability = result['probability'] ?? 0.0;
        return "예측: $prediction (${(probability * 100).toStringAsFixed(1)}%)";
      } else {
        print("❌ 서버 오류: ${response.statusCode}");
        print("❌ 서버 응답 본문: ${response.body}");
        return "서버 오류: ${response.statusCode}\n${response.body}";
      }
    } catch (e) {
      return "예측 실패: $e";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          height: 200,
          child: LineChart(LineChartData(
            lineBarsData: [
              LineChartBarData(
                spots: List.generate(_data.length, (i) => FlSpot(i.toDouble(), _data[i])),
                isCurved: true,
                color: Colors.deepPurple,
                dotData: FlDotData(show: false),
                belowBarData: BarAreaData(show: false),
              )
            ],
            titlesData: FlTitlesData(show: false),
            borderData: FlBorderData(show: false),
            gridData: FlGridData(show: false),
          )),
        ),
        ElevatedButton.icon(
          onPressed: _toggleStreaming,
          icon: Icon(_isStreaming ? Icons.pause : Icons.play_arrow),
          label: Text(_isStreaming ? "중지" : "예측 시작"),
        ),
      ],
    );
  }
}
