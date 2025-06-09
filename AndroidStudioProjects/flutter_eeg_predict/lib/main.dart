import 'package:flutter/material.dart';
import 'eeg_stream_graph.dart'; // 예측 그래프 위젯 경로는 프로젝트에 맞게 수정

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          title: const Text('EEG 실시간 예측'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: EEGStreamGraph(
            onPrediction: (result) {
              print('🔥 최종 예측 결과: $result');
            },
          ),
        ),
      ),
    );
  }
}
