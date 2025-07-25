# File: dashboard-pendataan-desa/auth_utils.py

import streamlit as st
import requests
import time
from core_utils import init_firebase # Kita tetap butuh ini untuk cek role di Firestore

# --- Fungsi untuk Cek Peran Admin di Firestore ---
def check_admin_role(uid):
    """Memeriksa apakah UID pengguna memiliki peran 'admin' di Firestore."""
    try:
        db_admin = init_firebase()
        if not db_admin:
            st.error("Koneksi admin Firestore tidak tersedia untuk verifikasi peran.")
            return False
            
        user_doc = db_admin.collection('users').document(uid).get()
        if user_doc.exists and user_doc.to_dict().get('role') == 'admin':
            return True
        return False
    except Exception as e:
        st.error(f"Terjadi kesalahan saat verifikasi peran: {e}")
        return False

# --- Fungsi Login Utama (Menggunakan Requests) ---
def login_admin(email, password):
    """Mencoba login via Firebase REST API, memverifikasi peran, dan mengatur session state."""
    try:
        api_key = st.secrets["firebase_client_config"]["apiKey"]
        rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status() # Akan melempar error jika status code bukan 2xx
        
        user_data = response.json()
        uid = user_data.get('localId')
        
        # Cek apakah pengguna adalah admin
        if uid and check_admin_role(uid):
            st.session_state.logged_in = True
            st.session_state.user_email = user_data.get('email')
            st.toast(f"Login berhasil! Selamat datang, {st.session_state.user_email}.", icon="🎉")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Login Gagal: Anda tidak memiliki hak akses admin.")
            
    except requests.exceptions.HTTPError as e:
        st.error("Login Gagal: Email atau password salah.")
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")

# --- Fungsi Logout ---
def logout():
    """Membersihkan session state untuk logout."""
    if 'logged_in' in st.session_state:
        del st.session_state.logged_in
    if 'user_email' in st.session_state:
        del st.session_state.user_email
    st.toast("Anda telah logout.", icon="👋")
    time.sleep(1)
    st.rerun()