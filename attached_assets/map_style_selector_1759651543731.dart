// lib/widgets/map_style_selector.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/map_style_provider.dart';

class MapStyleSelector extends ConsumerWidget {
  const MapStyleSelector({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final currentStyle = ref.watch(mapStyleProvider);

    return Container(
      margin: const EdgeInsets.only(top: 80, right: 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Botón principal que muestra el estilo actual
          Container(
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.8),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: Colors.cyan.withOpacity(0.5),
                width: 1,
              ),
            ),
            child: IconButton(
              onPressed: () => _showStyleMenu(context, ref),
              icon: Icon(
                MapStyleNotifier.getStyleData(currentStyle).icon,
                color: Colors.cyan,
                size: 24,
              ),
            ),
          ),
          const SizedBox(height: 8),
          // Indicador del estilo actual
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.8),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: Colors.cyan.withOpacity(0.3),
                width: 1,
              ),
            ),
            child: Text(
              MapStyleNotifier.getStyleData(currentStyle).name,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
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
