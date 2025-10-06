// lib/providers/map_style_provider.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

enum MapStyle {
  dark,
  light,
  colorful,
  satellite,
  terrain,
  vintage,
}

class MapStyleData {
  final String name;
  final String urlTemplate;
  final List<String> subdomains;
  final Color markerColor;
  final Color borderColor;
  final String description;
  final IconData icon;

  const MapStyleData({
    required this.name,
    required this.urlTemplate,
    required this.subdomains,
    required this.markerColor,
    required this.borderColor,
    required this.description,
    required this.icon,
  });
}

class MapStyleNotifier extends StateNotifier<MapStyle> {
  MapStyleNotifier() : super(MapStyle.dark);

  void changeStyle(MapStyle newStyle) {
    state = newStyle;
  }

  MapStyleData get currentStyleData => getStyleData(state);

  static MapStyleData getStyleData(MapStyle style) {
    switch (style) {
      case MapStyle.dark:
        return const MapStyleData(
          name: 'Espacial Oscuro',
          urlTemplate:
              'https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
          subdomains: ['a', 'b', 'c', 'd'],
          markerColor: Colors.cyan,
          borderColor: Colors.cyan,
          description: 'Mapa oscuro perfecto para el espacio',
          icon: Icons.dark_mode,
        );

      case MapStyle.light:
        return const MapStyleData(
          name: 'Clásico Claro',
          urlTemplate:
              'https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
          subdomains: ['a', 'b', 'c', 'd'],
          markerColor: Colors.red,
          borderColor: Colors.blue,
          description: 'Mapa claro tradicional',
          icon: Icons.light_mode,
        );

      case MapStyle.colorful:
        return const MapStyleData(
          name: 'Países Coloridos',
          urlTemplate:
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
          subdomains: [],
          markerColor: Colors.purple,
          borderColor: Colors.deepPurple,
          description: 'Mapa con países en colores vibrantes',
          icon: Icons.palette,
        );

      case MapStyle.satellite:
        return const MapStyleData(
          name: 'Vista Satélite',
          urlTemplate:
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          subdomains: [],
          markerColor: Colors.yellow,
          borderColor: Colors.orange,
          description: 'Imágenes reales desde satélite',
          icon: Icons.satellite_alt,
        );

      case MapStyle.terrain:
        return const MapStyleData(
          name: 'Relieve Terrestre',
          urlTemplate:
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}',
          subdomains: [],
          markerColor: Colors.green,
          borderColor: Colors.teal,
          description: 'Relieve y topografía mundial',
          icon: Icons.terrain,
        );

      case MapStyle.vintage:
        return const MapStyleData(
          name: 'Estilo Vintage',
          urlTemplate:
              'https://cartodb-basemaps-{s}.global.ssl.fastly.net/rastertiles/voyager/{z}/{x}/{y}.png',
          subdomains: ['a', 'b', 'c', 'd'],
          markerColor: Colors.amber,
          borderColor: Colors.brown,
          description: 'Mapa con estilo retro elegante',
          icon: Icons.camera_alt,
        );
    }
  }
}

// Provider principal
final mapStyleProvider =
    StateNotifierProvider<MapStyleNotifier, MapStyle>((ref) {
  return MapStyleNotifier();
});

// Provider para obtener los datos del estilo actual
final currentMapStyleDataProvider = Provider<MapStyleData>((ref) {
  final currentStyle = ref.watch(mapStyleProvider);
  return MapStyleNotifier.getStyleData(currentStyle);
});
