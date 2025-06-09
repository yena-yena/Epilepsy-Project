import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'stream_service.dart';

class EEGStreamGraph extends StatefulWidget {
  final Function(String)? onPrediction;

  const EEGStreamGraph({super.key, this.onPrediction});

  @override
  State<EEGStreamGraph> createState() => _EEGStreamGraphState();
}

class _EEGStreamGraphState extends State<EEGStreamGraph> {
  final int _windowSize = 80;
  final int _channelCount = 8;
  final int _maxLength = 200;

  List<List<double>> _multiData = List.generate(8, (_) => []);
  Timer? _timer;
  bool _isStreaming = false;
  List<List<double>> _buffer = [];
  String _predictionText = "예측 대기 중...";
  List<String> _statusLog = [];

  final List<Color> _colors = [
    Colors.deepPurple, Colors.orange, Colors.green, Colors.red,
    Colors.blue, Colors.brown, Colors.pink, Colors.teal
  ];

  void _toggleStreaming() {
    if (_isStreaming) {
      _timer?.cancel();
      setState(() {
        _isStreaming = false;
        _predictionText = "예측이 중지되었습니다.";
        _buffer.clear();
        _multiData = List.generate(_channelCount, (_) => []);
        // ❗ 예측 로그는 유지하고 새로운 건 안 뜨게
      });
    } else {
      setState(() {
        _isStreaming = true;
        _predictionText = "예측 대기 중...";
        _buffer.clear();
        _multiData = List.generate(_channelCount, (_) => []);
        _statusLog.clear();
      });
      Future.delayed(const Duration(milliseconds: 100), () {
        if (mounted) _startStreaming();
      });
    }
  }

  void _startStreaming() {
    _timer = Timer.periodic(const Duration(seconds: 1), (_) async {
      if (!mounted) {
        _timer?.cancel();
        return;
      }

      try {
        final eegInfo = await StreamService.fetchNextEEG();
        if (eegInfo == null || eegInfo['data'] == null) {
          if (mounted) {
            setState(() {
              _predictionText = "스트림 서버 연결 실패";
            });
            _toggleStreaming();
          }
          return;
        }

        final List<List<double>> newDataChunk = (eegInfo['data'] as List)
            .map<List<double>>((row) =>
            (row as List).map<double>((e) => (e as num).toDouble()).toList())
            .toList();

        if (newDataChunk.isEmpty || newDataChunk[0].length != _channelCount) {
          return;
        }

        // 🧠 부드럽게 표시: 한 포인트씩 순차적으로 그리기
        for (int t = 0; t < newDataChunk.length; t++) {
          for (int ch = 0; ch < _channelCount; ch++) {
            _multiData[ch].add(newDataChunk[t][ch] * 10000);
            if (_multiData[ch].length > _maxLength) {
              _multiData[ch].removeAt(0);
            }
          }
          await Future.delayed(const Duration(milliseconds: 20));
        }

        _buffer.addAll(newDataChunk);

        if (_buffer.length >= _windowSize) {
          final dataForPrediction = List.generate(
            _channelCount,
                (channelIndex) => List.generate(
              _windowSize,
                  (timeIndex) => _buffer[_buffer.length - _windowSize + timeIndex][channelIndex],
            ),
          );

          final prediction = await _sendPredictionRequest(dataForPrediction);

          if (mounted && _isStreaming) {
            setState(() {
              _predictionText = prediction;
              _statusLog.add(prediction);
              if (_statusLog.length > 5) {
                _statusLog.removeAt(0);
              }
            });
          }

          if (_buffer.length > _maxLength) {
            _buffer = _buffer.sublist(_buffer.length - _maxLength);
          }
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _predictionText = "오류 발생: $e";
          });
        }
      }
    });
  }

  Future<String> _sendPredictionRequest(List<List<double>> data) async {
    if (data.length != _channelCount || data[0].length != _windowSize) {
      final errorMsg = "❌ 예측 요청 전 데이터 모양 오류! shape: (${data.length}, ${data[0].length})";
      widget.onPrediction?.call(errorMsg);
      return errorMsg;
    }

    const url = 'http://10.0.2.2:8001/predict';
    try {
      final payload = {"data": data};

      final response = await http.post(
        Uri.parse(url),
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: jsonEncode(payload),
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (result.containsKey('error')) {
          final errorMsg = "서버 오류: ${result['error']}";
          widget.onPrediction?.call(errorMsg);
          return errorMsg;
        }

        final probability = result['probability'] ?? 0.0;
        final percentage = (probability * 100).toStringAsFixed(1);

        String predictionResult = "";
        if (probability > 0.7) {
          predictionResult = "⚠️ 발작 가능성 높음 ($percentage%)";
        } else if (probability > 0.4) {
          predictionResult = "🟡 발작 경고 ($percentage%)";
        } else {
          predictionResult = "✅ 정상 ($percentage%)";
        }

        widget.onPrediction?.call(predictionResult);
        return predictionResult;
      } else {
        final error = "서버 오류: ${response.statusCode}\n${response.body}";
        widget.onPrediction?.call(error);
        return error;
      }
    } catch (e) {
      final error = "연결 실패: $e";
      widget.onPrediction?.call(error);
      return error;
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final allValues = _multiData.expand((list) => list).toList();
    final minY = allValues.isEmpty ? -2.0 : allValues.reduce((a, b) => a < b ? a : b);
    final maxY = allValues.isEmpty ? 2.0 : allValues.reduce((a, b) => a > b ? a : b);

    return Column(
      children: [
        Container(
          height: 250,
          padding: const EdgeInsets.all(16),
          child: allValues.isEmpty
              ? const Center(
            child: Text(
              "데이터를 기다리는 중...",
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
          )
              : LineChart(
            LineChartData(
              lineBarsData: List.generate(_channelCount, (ch) {
                return LineChartBarData(
                  spots: List.generate(
                    _multiData[ch].length,
                        (i) => FlSpot(i.toDouble(), _multiData[ch][i]),
                  ),
                  isCurved: false,
                  color: _colors[ch % _colors.length],
                  barWidth: 2,
                  dotData: const FlDotData(show: false),
                  belowBarData: BarAreaData(show: false),
                );
              }),
              titlesData: const FlTitlesData(show: false),
              borderData: FlBorderData(
                show: true,
                border: Border.all(color: Colors.grey.withAlpha(77)),
              ),
              gridData: const FlGridData(show: true),
              minY: minY,
              maxY: maxY,
            ),
          ),
        ),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(12),
          margin: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            color: Colors.grey.withAlpha(25),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.grey.withAlpha(77)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: _statusLog.reversed.map((status) {
              Color color;
              if (status.contains("높음")) {
                color = Colors.red;
              } else if (status.contains("경고")) {
                color = Colors.orange;
              } else if (status.contains("정상")) {
                color = Colors.green;
              } else {
                color = Colors.black;
              }
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Text(
                  status,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
              );
            }).toList(),
          ),
        ),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          onPressed: _toggleStreaming,
          icon: Icon(_isStreaming ? Icons.pause : Icons.play_arrow),
          label: Text(_isStreaming ? "중지" : "예측 시작"),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            backgroundColor: _isStreaming ? Colors.red : Colors.green,
            foregroundColor: Colors.white,
          ),
        ),
      ],
    );
  }
}
