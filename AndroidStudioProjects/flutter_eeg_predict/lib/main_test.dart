import 'package:flutter/material.dart';
import 'package:flutter_eeg_predict/eeg_stream_test.dart';

void main() {
  runApp(const EEGTestApp());
}

class EEGTestApp extends StatelessWidget {
  const EEGTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EEG 실시간 예측 테스트',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: Scaffold(
        appBar: AppBar(
          title: const Text('EEG 실시간 예측 테스트'),
        ),
        body: const Padding(
          padding: EdgeInsets.all(16.0),
          child: EEGStreamTestGraph(), // ✅ 오타 수정
        ),
      ),
    );
  }
}
