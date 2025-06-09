import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class EEGPredictApp extends StatefulWidget {
  @override
  _EEGPredictAppState createState() => _EEGPredictAppState();
}

class _EEGPredictAppState extends State<EEGPredictApp> {
  double? probability;
  int? prediction;
  bool isLoading = false;

  // 예시 EEG 데이터 (8x38)
  List<List<double>> sampleEEGData = List.generate(
    8,
        (_) => List.generate(38, (index) => (index % 5 == 0 ? 1.0 : 0.0)),
  );

  Future<void> fetchPrediction() async {
    setState(() => isLoading = true);

    final url = Uri.parse("http://localhost:8000/predict");
    try {
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
          body: jsonEncode({ "data": sampleEEGData }),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        setState(() {
          prediction = result['prediction'];
          probability = result['probability'];
        });
      } else {
        print("서버 오류: ${response.statusCode}");
      }
    } catch (e) {
      print("요청 실패: $e");
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('EEG Seizure Predictor')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                '예측 결과',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 20),
              if (isLoading) CircularProgressIndicator(),
              if (!isLoading && probability != null)
                Column(
                  children: [
                    Text('발작 확률: ${(probability! * 100).toStringAsFixed(2)}%',
                        style: TextStyle(fontSize: 20)),
                    Text('결과: ${prediction == 1 ? "발작" : "비발작"}',
                        style: TextStyle(fontSize: 20)),
                  ],
                ),
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: fetchPrediction,
                child: Text('예측 요청'),
              )
            ],
          ),
        ),
      ),
    );
  }
}
