// lib/utils/utils.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class Utils {
  /// Formatea una fecha en formato DD/MM/AA
  /// Ejemplo: 04/10/25
  static String formatDate(DateTime date) {
    return DateFormat('dd/MM/yy').format(date);
  }

  /// Formatea una hora en formato 24:00
  /// Ejemplo: 14:30
  static String formatTime(DateTime dateTime) {
    return DateFormat('HH:mm').format(dateTime);
  }

  /// Formatea fecha y hora completa en formato DD/MM/AA HH:mm
  /// Ejemplo: 04/10/25 14:30
  static String formatDateTime(DateTime dateTime) {
    return '${formatDate(dateTime)} ${formatTime(dateTime)}';
  }

  /// Formatea solo la fecha en formato completo DD/MM/AAAA
  /// Ejemplo: 04/10/2025
  static String formatDateFull(DateTime date) {
    return DateFormat('dd/MM/yyyy').format(date);
  }

  /// Combina fecha y hora por separado en un solo DateTime
  static DateTime combineDateTime(DateTime date, TimeOfDay time) {
    return DateTime(
      date.year,
      date.month,
      date.day,
      time.hour,
      time.minute,
    );
  }
}
