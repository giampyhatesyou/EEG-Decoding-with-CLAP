#!/usr/bin/env bash
# =============================================================================
# run_all_cheap.sh — in sequenza SOLO gli esperimenti che NON spendono sguardi
# e NON richiedono GPU.
#
# Cosa gira, e perche' e' gratis:
#   exp13  separabilita' solo-audio  — nessun file EEG viene mai aperto
#   exp07  sweep own-vs-other        — solo segmenti SOLO held-out (assert raw_solos)
#   exp08  flux su own-vs-other      — idem
#   exp14  MFCC/mel-64 su own-vs-other — idem
#
# Cosa NON gira, e va lanciato a mano sapendo cosa costa:
#   exp09 · exp11 · exp12 · exp15 · braccio A · braccio D  -> decidono sui DUO.
#   I 309 duo sono tutti gia' spesi: rigirarli non apre materiale nuovo, ma ogni
#   numero sui duo e' ESPLORATIVO PER COSTRUZIONE e va etichettato cosi'.
#   🔒 I TRIO (92 stereo + 93 mono) non li tocca nessuno script di questa cartella.
#
# COSTO TOTALE: ~25-50 min CPU, dominato dall'Exp. 7 (C5 e C7 girano a
#   --target_fs 256, ~6 GB l'una). Vedi le stime in testa a ogni script.
#
# Ogni script si ferma da solo se il suo canarino non passa, e `set -e` qui ferma
# la sequenza: un canarino rosso NON deve produrre i numeri successivi.
#
#   bash scripts/replicate/run_all_cheap.sh
#   ONLY="exp13 exp14" bash scripts/replicate/run_all_cheap.sh   # un sottoinsieme
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ONLY=${ONLY:-"exp13 exp07 exp08 exp14"}

# case e non un array associativo: il bash di macOS e' il 3.2, che non ha `declare -A`.
script_for() {
  case "$1" in
    exp13) echo exp13_stem_separability.sh ;;
    exp07) echo exp07_ovo_config_sweep.sh ;;
    exp08) echo exp08_flux_ovo.sh ;;
    exp14) echo exp14_mfcc_tracking.sh ;;
    *)     echo "" ;;
  esac
}

for k in $ONLY; do
  s=$(script_for "$k")
  [ -n "$s" ] || { echo "sconosciuto: $k (validi: exp13 exp07 exp08 exp14)"; exit 2; }
  echo
  echo "############################################################"
  echo "# $k  ->  $s   $(date '+%H:%M:%S')"
  echo "############################################################"
  bash "$HERE/$s"
done

echo
echo "############################################################"
echo "# tutti gli esperimenti gratuiti sono passati  $(date '+%H:%M:%S')"
echo "# nessuno sguardo speso · nessuna GPU · trio non toccati"
echo "############################################################"
