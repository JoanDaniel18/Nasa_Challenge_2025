// lib/screens/map_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../providers/selection_provider.dart';
import '../providers/map_style_provider.dart';
import '../utils/utils.dart';
import 'dart:math' as math;

class MapScreen extends ConsumerWidget {
  const MapScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selection = ref.watch(selectionProvider);
    final mapStyleData = ref.watch(currentMapStyleDataProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Weather'),
        backgroundColor: const Color(0xFF0D1B2A),
        foregroundColor: Colors.white,
        actions: [
          // Selector de estilos de mapa en el AppBar
          Container(
            margin: const EdgeInsets.only(right: 8),
            child: IconButton(
              onPressed: () => _showStyleMenu(context, ref),
              icon: Icon(
                mapStyleData.icon,
                color: mapStyleData.markerColor,
                size: 28,
              ),
              tooltip: 'Cambiar estilo: ${mapStyleData.name}',
            ),
          ),
        ],
      ),
      backgroundColor: const Color(0xFF000000),
      body: Container(
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment.center,
            radius: 1.0,
            colors: [
              Color(0xFF0D1B2A),
              Color(0xFF000000),
            ],
          ),
        ),
        child: Column(
          children: [
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  gradient: const RadialGradient(
                    center: Alignment.center,
                    radius: 1.5,
                    colors: [
                      Color(0xFF0D1B2A),
                      Color(0xFF1B263B),
                      Color(0xFF000000),
                    ],
                  ),
                  border:
                      Border.all(color: Colors.cyan.withOpacity(0.3), width: 2),
                  borderRadius: BorderRadius.circular(20),
                ),
                margin: const EdgeInsets.all(16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: Stack(
                    children: [
                      // Estrellas de fondo
                      ...List.generate(50, (index) => _buildStar(index)),

                      // Mapa con tema espacial
                      FlutterMap(
                        options: MapOptions(
                          initialCenter: const LatLng(20, 0),
                          initialZoom: 2.0,
                          minZoom: 1,
                          maxZoom: 18,
                          backgroundColor: Colors.transparent,
                          onTap: (tapPos, point) {
                            ref.read(selectionProvider.notifier).state =
                                selection.copyWith(coordinates: point);
                          },
                        ),
                        children: [
                          // Tile layer con estilo dinámico
                          TileLayer(
                            urlTemplate: mapStyleData.urlTemplate,
                            subdomains: mapStyleData.subdomains,
                            additionalOptions: const {
                              'attribution': '© Map Provider',
                            },
                          ),

                          // Marcador con efecto neón
                          if (selection.coordinates != null)
                            MarkerLayer(
                              markers: [
                                Marker(
                                  point: selection.coordinates!,
                                  width: 60,
                                  height: 60,
                                  child: Stack(
                                    alignment: Alignment.center,
                                    children: [
                                      // Círculo pulsante exterior
                                      Container(
                                        width: 40,
                                        height: 40,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          border: Border.all(
                                            color: mapStyleData.markerColor
                                                .withOpacity(0.5),
                                            width: 2,
                                          ),
                                        ),
                                      ),
                                      // Círculo medio
                                      Container(
                                        width: 25,
                                        height: 25,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: mapStyleData.markerColor
                                              .withOpacity(0.3),
                                          border: Border.all(
                                            color: mapStyleData.markerColor,
                                            width: 1,
                                          ),
                                        ),
                                      ),
                                      // Punto central brillante
                                      Container(
                                        width: 12,
                                        height: 12,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: mapStyleData.markerColor,
                                          boxShadow: [
                                            BoxShadow(
                                              color: mapStyleData.markerColor,
                                              blurRadius: 8,
                                              spreadRadius: 2,
                                            ),
                                          ],
                                        ),
                                      ),
                                      // Icono central
                                      const Icon(
                                        Icons.my_location,
                                        color: Colors.white,
                                        size: 16,
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                        ],
                      ),

                      // Overlay de coordenadas
                      if (selection.coordinates != null)
                        Positioned(
                          top: 16,
                          left: 16,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 8,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.8),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color:
                                    mapStyleData.borderColor.withOpacity(0.5),
                                width: 1,
                              ),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  '📍 COORDENADAS',
                                  style: TextStyle(
                                    color: mapStyleData.markerColor,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'LAT: ${selection.coordinates!.latitude.toStringAsFixed(4)}°',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontFamily: 'monospace',
                                  ),
                                ),
                                Text(
                                  'LNG: ${selection.coordinates!.longitude.toStringAsFixed(4)}°',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontFamily: 'monospace',
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
            ListTile(
              tileColor: Colors.black.withOpacity(0.5),
              title: Text(
                selection.datetime != null
                    ? 'Fecha y hora: ${Utils.formatDateTime(selection.datetime!)}'
                    : 'Selecciona fecha y hora',
                style: const TextStyle(color: Colors.white),
              ),
              trailing: const Icon(Icons.calendar_today, color: Colors.cyan),
              onTap: () async {
                DateTime? picked = await showDatePicker(
                  context: context,
                  initialDate: selection.datetime ?? DateTime.now(),
                  firstDate: DateTime(2000),
                  lastDate: DateTime(2100),
                  builder: (context, child) {
                    return Theme(
                      data: Theme.of(context).copyWith(
                        colorScheme: const ColorScheme.dark(
                          primary: Colors.cyan,
                          onPrimary: Colors.black,
                          surface: Color(0xFF1B263B),
                          onSurface: Colors.white,
                        ),
                      ),
                      child: child!,
                    );
                  },
                );

                if (picked != null) {
                  TimeOfDay? time = await showTimePicker(
                    context: context,
                    initialTime: TimeOfDay.fromDateTime(
                        selection.datetime ?? DateTime.now()),
                    builder: (context, child) {
                      return Theme(
                        data: Theme.of(context).copyWith(
                          colorScheme: const ColorScheme.dark(
                            primary: Colors.cyan,
                            onPrimary: Colors.black,
                            surface: Color(0xFF1B263B),
                            onSurface: Colors.white,
                          ),
                        ),
                        child: child!,
                      );
                    },
                  );

                  if (time != null) {
                    DateTime finalDateTime =
                        Utils.combineDateTime(picked, time);

                    ref.read(selectionProvider.notifier).state =
                        selection.copyWith(datetime: finalDateTime);
                  }
                }
              },
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.cyan,
                  foregroundColor: Colors.black,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  elevation: 8,
                  shadowColor: Colors.cyan.withOpacity(0.5),
                ),
                onPressed: () async {
                  try {
                    final sendSelection = ref.read(sendSelectionProvider);
                    final result = await sendSelection(selection);
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('🚀 Datos enviados: $result'),
                        backgroundColor: Colors.green.shade800,
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  } catch (e) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('❌ Error: $e'),
                        backgroundColor: Colors.red.shade800,
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  }
                },
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.rocket_launch),
                    SizedBox(width: 8),
                    Text('ENVIAR DATOS',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStar(int index) {
    final random = math.Random(index);
    return Positioned(
      left: random.nextDouble() * 400,
      top: random.nextDouble() * 300,
      child: Container(
        width: random.nextDouble() * 3 + 1,
        height: random.nextDouble() * 3 + 1,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(random.nextDouble() * 0.8 + 0.2),
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: Colors.white.withOpacity(0.5),
              blurRadius: 2,
            ),
          ],
        ),
      ),
    );
  }

  void _showStyleMenu(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        child: Container(
          width: 320,
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.9),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.cyan.withOpacity(0.5),
              width: 2,
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.cyan.withOpacity(0.2),
                      Colors.transparent,
                    ],
                  ),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(18),
                    topRight: Radius.circular(18),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.map, color: Colors.cyan),
                    const SizedBox(width: 12),
                    const Text(
                      'Estilos de Mapa',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Spacer(),
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.close, color: Colors.cyan),
                    ),
                  ],
                ),
              ),

              // Grid de estilos
              Padding(
                padding: const EdgeInsets.all(16),
                child: GridView.builder(
                  shrinkWrap: true,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    childAspectRatio: 1.2,
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                  ),
                  itemCount: MapStyle.values.length,
                  itemBuilder: (context, index) {
                    final style = MapStyle.values[index];
                    final styleData = MapStyleNotifier.getStyleData(style);
                    final isSelected = ref.watch(mapStyleProvider) == style;

                    return GestureDetector(
                      onTap: () {
                        ref.read(mapStyleProvider.notifier).changeStyle(style);
                        Navigator.pop(context);
                      },
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? styleData.markerColor.withOpacity(0.2)
                              : Colors.grey.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: isSelected
                                ? styleData.markerColor
                                : Colors.grey.withOpacity(0.3),
                            width: isSelected ? 2 : 1,
                          ),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              styleData.icon,
                              color: isSelected
                                  ? styleData.markerColor
                                  : Colors.white,
                              size: 32,
                            ),
                            const SizedBox(height: 8),
                            Text(
                              styleData.name,
                              style: TextStyle(
                                color: isSelected
                                    ? styleData.markerColor
                                    : Colors.white,
                                fontSize: 12,
                                fontWeight: isSelected
                                    ? FontWeight.bold
                                    : FontWeight.normal,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              styleData.description,
                              style: TextStyle(
                                color: Colors.grey.shade400,
                                fontSize: 10,
                              ),
                              textAlign: TextAlign.center,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
