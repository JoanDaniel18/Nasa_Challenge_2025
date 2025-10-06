// lib/widgets/globe_widget.dart
import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';
import 'dart:math' as math;

class GlobeWidget extends StatefulWidget {
  final LatLng? selectedCoordinates;
  final Function(LatLng) onTap;
  final double radius;

  const GlobeWidget({
    super.key,
    this.selectedCoordinates,
    required this.onTap,
    this.radius = 150.0,
  });

  @override
  State<GlobeWidget> createState() => _GlobeWidgetState();
}

class _GlobeWidgetState extends State<GlobeWidget>
    with TickerProviderStateMixin {
  late AnimationController _rotationController;
  double _rotation = 0;
  double _tilt = 0.3; // Inclinación del globo
  Offset? _dragStart;

  @override
  void initState() {
    super.initState();
    _rotationController = AnimationController(
      duration: const Duration(seconds: 30),
      vsync: this,
    )..repeat();

    _rotationController.addListener(() {
      setState(() {
        _rotation += 0.01; // Rotación automática lenta
      });
    });
  }

  @override
  void dispose() {
    _rotationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: RadialGradient(
          center: Alignment.center,
          radius: 1.5,
          colors: [
            Color(0xFF0D1B2A), // Azul espacio profundo
            Color(0xFF1B263B), // Azul mediano
            Color(0xFF415A77), // Azul gris
            Color(0xFF000000), // Negro espacio
          ],
          stops: [0.0, 0.3, 0.7, 1.0],
        ),
      ),
      child: Stack(
        children: [
          // Estrellas de fondo
          ...List.generate(100, (index) => _buildStar(index)),

          // Globo terráqueo
          Center(
            child: GestureDetector(
              onPanStart: (details) {
                _rotationController.stop();
                _dragStart = details.localPosition;
              },
              onPanUpdate: (details) {
                if (_dragStart != null) {
                  final delta = details.localPosition - _dragStart!;
                  setState(() {
                    _rotation += delta.dx * 0.01;
                    _tilt = (_tilt + delta.dy * 0.005).clamp(-0.5, 0.5);
                  });
                  _dragStart = details.localPosition;
                }
              },
              onPanEnd: (details) {
                _rotationController.repeat();
              },
              onTapUp: (details) {
                _handleTap(details.localPosition);
              },
              child: Transform(
                alignment: Alignment.center,
                transform: Matrix4.identity()
                  ..setEntry(3, 2, 0.001) // Perspectiva
                  ..rotateX(_tilt)
                  ..rotateY(_rotation),
                child: Container(
                  width: widget.radius * 2,
                  height: widget.radius * 2,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      center: const Alignment(-0.4, -0.4), // Luz del sol
                      radius: 1.2,
                      colors: [
                        Colors.lightBlue.shade100, // Iluminado
                        Colors.blue.shade300,
                        Colors.blue.shade600,
                        Colors.blue.shade800,
                        Colors.indigo.shade900,
                        Colors.black, // Sombra
                      ],
                      stops: const [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.cyan.withOpacity(0.3),
                        blurRadius: 30,
                        spreadRadius: 10,
                      ),
                      BoxShadow(
                        color: Colors.black.withOpacity(0.6),
                        blurRadius: 20,
                        spreadRadius: 5,
                        offset: const Offset(15, 15),
                      ),
                    ],
                  ),
                  child: Stack(
                    children: [
                      // Continentes simplificados
                      CustomPaint(
                        size: Size(widget.radius * 2, widget.radius * 2),
                        painter: EarthPainter(_rotation, _tilt),
                      ),

                      // Marcador de selección
                      if (widget.selectedCoordinates != null)
                        _buildMarker(widget.selectedCoordinates!),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStar(int index) {
    final random = math.Random(index);
    return Positioned(
      left: random.nextDouble() * 400,
      top: random.nextDouble() * 600,
      child: Container(
        width: random.nextDouble() * 3 + 1,
        height: random.nextDouble() * 3 + 1,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(random.nextDouble() * 0.8 + 0.2),
          shape: BoxShape.circle,
        ),
      ),
    );
  }

  Widget _buildMarker(LatLng coordinates) {
    final pos = _projectCoordinates(coordinates);
    if (pos == null) return const SizedBox.shrink();

    return Positioned(
      left: widget.radius + pos.dx - 10,
      top: widget.radius + pos.dy - 10,
      child: Container(
        width: 20,
        height: 20,
        decoration: BoxDecoration(
          color: Colors.red,
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 2),
          boxShadow: [
            BoxShadow(
              color: Colors.red.withOpacity(0.5),
              blurRadius: 10,
              spreadRadius: 2,
            ),
          ],
        ),
        child: const Icon(
          Icons.location_on,
          color: Colors.white,
          size: 12,
        ),
      ),
    );
  }

  Offset? _projectCoordinates(LatLng coordinates) {
    // Proyección 3D simple de coordenadas geográficas
    final lat = coordinates.latitude * math.pi / 180;
    final lng =
        (coordinates.longitude + _rotation * 180 / math.pi) * math.pi / 180;

    final x = widget.radius * math.cos(lat) * math.cos(lng);
    final y = widget.radius * math.cos(lat) * math.sin(lng);
    final z = widget.radius * math.sin(lat);

    // Solo mostrar si está en el frente del globo
    if (y < 0) return null;

    return Offset(x * math.cos(_tilt) - z * math.sin(_tilt),
        z * math.cos(_tilt) + x * math.sin(_tilt));
  }

  void _handleTap(Offset localPosition) {
    // Convertir posición de tap a coordenadas geográficas (simplificado)
    final center = Offset(widget.radius, widget.radius);
    final relative = localPosition - center;
    final distance = relative.distance;

    if (distance <= widget.radius) {
      // Proyección inversa simplificada
      final lat = math.asin(relative.dy / widget.radius) * 180 / math.pi;
      final lng = math.atan2(relative.dx, widget.radius) * 180 / math.pi -
          _rotation * 180 / math.pi;

      final coordinates = LatLng(lat.clamp(-90, 90), lng);
      widget.onTap(coordinates);
    }
  }
}

class EarthPainter extends CustomPainter {
  final double rotation;
  final double tilt;

  EarthPainter(this.rotation, this.tilt);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green.shade600
      ..style = PaintingStyle.fill;

    final radius = size.width / 2;
    final center = Offset(radius, radius);

    // Dibujar continentes simplificados (formas básicas que rotan)
    _drawContinent(
        canvas, paint, center, radius, 0, 40, 60, 20); // Europa/África
    _drawContinent(canvas, paint, center, radius, -100, 35, 80, 30); // América
    _drawContinent(
        canvas, paint, center, radius, 120, -20, 70, 40); // Asia/Oceanía
  }

  void _drawContinent(Canvas canvas, Paint paint, Offset center, double radius,
      double baseAngle, double baseLat, double width, double height) {
    final angle = baseAngle + rotation * 180 / math.pi;
    final lat = baseLat * math.pi / 180;
    final lng = angle * math.pi / 180;

    final x = radius * math.cos(lat) * math.cos(lng);
    final y = radius * math.cos(lat) * math.sin(lng);
    final z = radius * math.sin(lat);

    // Solo dibujar si está visible (frente del globo)
    if (y > -radius * 0.2) {
      final projectedX = x * math.cos(tilt) - z * math.sin(tilt);
      final projectedY = z * math.cos(tilt) + x * math.sin(tilt);

      final continentCenter = Offset(
        center.dx + projectedX,
        center.dy + projectedY,
      );

      final scale = (y + radius) / (2 * radius); // Efecto de perspectiva
      final rect = Rect.fromCenter(
        center: continentCenter,
        width: width * scale,
        height: height * scale,
      );

      canvas.drawOval(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
