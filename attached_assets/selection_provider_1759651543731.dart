// lib/providers/selection_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class Selection {
  final LatLng? coordinates;
  final DateTime? datetime;

  Selection({this.coordinates, this.datetime});

  Selection copyWith({LatLng? coordinates, DateTime? datetime}) {
    return Selection(
      coordinates: coordinates ?? this.coordinates,
      datetime: datetime ?? this.datetime,
    );
  }
}

// Estado de la selección
final selectionProvider = StateProvider<Selection>((ref) => Selection());

// Función para enviar los datos al backend
final sendSelectionProvider = Provider((ref) {
  return (Selection selection) async {
    if (selection.coordinates == null || selection.datetime == null) {
      throw Exception('Faltan coordenadas o fecha/hora');
    }

    final body = jsonEncode({
      'latitude': selection.coordinates!.latitude,
      'longitude': selection.coordinates!.longitude,
      'datetime': selection.datetime!.toIso8601String(),
    });

    final response = await http.post(
      Uri.parse('https://tuapi.com/weather'), // Cambiar por tu endpoint real
      headers: {'Content-Type': 'application/json'},
      body: body,
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Error al enviar datos: ${response.statusCode}');
    }
  };
});
