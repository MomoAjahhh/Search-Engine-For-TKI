import re
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import warnings
warnings.filterwarnings('ignore')

# KONFIGURASI

FILEPATH     = 'hasil_text_processing_vaksinasi.xlsx'
TOP_K        = 5
JUMLAH_BARIS = 75
MODEL_NAME   = 'indobenchmark/indobert-base-p1'
BATCH_SIZE   = 8

LABEL_SUMBER = {
    1 : 'Jurnal / Berita',
    2 : 'Media Sosial',
}


# MEMUAT DATA

def muat_dataset(filepath: str):
    df_raw = pd.read_excel(filepath)

    kandidat_asli = ['cleaned', 'sentence', 'text', 'content']
    kolom_asli = next((c for c in kandidat_asli if c in df_raw.columns), None)

    if not kolom_asli:
        raise ValueError(
            f"Kolom teks asli tidak ditemukan. Kolom tersedia: {list(df_raw.columns)}"
        )

    df = (df_raw
          .head(JUMLAH_BARIS)
          .dropna(subset=[kolom_asli])
          .reset_index(drop=True))
    df.insert(0, 'id_dok', range(1, len(df) + 1))

    print(f"Dataset dimuat: {len(df)} dokumen | Kolom teks: '{kolom_asli}'")
    return df, kolom_asli


#INDOBERT SEMANTIC SEARCH

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    
    token_embeddings = model_output.last_hidden_state                              
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings   = torch.sum(token_embeddings * mask_expanded, dim=1)          
    sum_mask         = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)             
    return sum_embeddings / sum_mask                                               


def ekstrak_embeddings(teks_list: list, tokenizer, model, device) -> np.ndarray:
    
    semua_emb = []

    for i in range(0, len(teks_list), BATCH_SIZE):
        batch_teks = teks_list[i : i + BATCH_SIZE]

        encoded = tokenizer(
            batch_teks,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        emb_batch = mean_pooling(output, encoded['attention_mask'])
        semua_emb.append(emb_batch.cpu().numpy())

    return np.vstack(semua_emb)


def cari_indobert(query: str, embeddings_dok: np.ndarray,
                  tokenizer, model, device, top_k: int) -> list:

    encoded_q = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    encoded_q = {k: v.to(device) for k, v in encoded_q.items()}

    with torch.no_grad():
        output_q = model(**encoded_q)

    emb_query   = mean_pooling(output_q, encoded_q['attention_mask']).cpu().numpy()
    skor_matrix = sk_cosine(emb_query, embeddings_dok)[0]                  

    indeks_sorted = np.argsort(skor_matrix)[::-1][:top_k]
    return [(int(idx), float(skor_matrix[idx])) for idx in indeks_sorted]


# UI UTAMA

def jalankan_ui(df, kolom_asli, embeddings_dokumen, tokenizer, model, device):

    BG         = "#DFDEDC"
    PANEL      = "#FFFFFF"
    MERAH      = "#9E3732"
    ABU        = "#6B7280"
    ABU_MUDA   = "#E5E7EB"
    HITAM      = "#3E3439"
    BIRU       = "#607EB1"

    FONT_JUDUL  = ("Segoe UI", 16, "bold")
    FONT_LABEL  = ("Segoe UI", 10, "bold")
    FONT_NORMAL = ("Segoe UI", 10)
    FONT_KECIL  = ("Segoe UI", 9)

    root = tk.Tk()
    root.title("Mesin Pencari Semantik — IndoBERT (Vaksinasi)")
    root.geometry("1100x680")
    root.configure(bg=BG)
    root.resizable(True, True)
    root.minsize(800, 460)

    frm_header = tk.Frame(root, bg=MERAH, padx=20, pady=12)
    frm_header.pack(fill="x")

    tk.Label(frm_header,
             text="Search Engine IR Moedern",
             font=FONT_JUDUL, bg=MERAH, fg="white").pack(side="left")

    tk.Label(frm_header,
             text=f"  |  {len(df)} dokumen  |  dim=768",
             font=("Segoe UI", 10), bg=MERAH, fg="#FFFFFF").pack(side="left", pady=2)

    frm_input = tk.Frame(root, bg=BG, padx=20, pady=12)
    frm_input.pack(fill="x")

    tk.Label(frm_input, text="Kata Kunci :", font=FONT_LABEL,
             bg=BG, fg=HITAM).grid(row=0, column=0, sticky="w", padx=(0, 8))

    var_query = tk.StringVar()
    entry = tk.Entry(frm_input, textvariable=var_query, font=FONT_NORMAL,
                     relief="flat", bg=PANEL, fg=HITAM,
                     highlightthickness=1, highlightbackground=ABU_MUDA,
                     highlightcolor=MERAH, width=60)
    entry.grid(row=0, column=1, ipady=7, sticky="ew")

    tk.Label(frm_input, text="  Top-K :", font=FONT_LABEL,
             bg=BG, fg=HITAM).grid(row=0, column=2, padx=(12, 4))

    var_topk = tk.IntVar(value=TOP_K)
    spin = tk.Spinbox(frm_input, from_=1, to=20, textvariable=var_topk,
                      width=4, font=FONT_NORMAL, relief="flat",
                      bg=PANEL, fg=HITAM, bd=0,
                      highlightthickness=1, highlightbackground=ABU_MUDA)
    spin.grid(row=0, column=3, ipady=6)

    btn_cari = tk.Button(frm_input, text="  Cari  ", font=FONT_LABEL,
                         bg=MERAH, fg="white", relief="flat",
                         activebackground=MERAH, activeforeground="white",
                         cursor="hand2", padx=14, pady=5)
    btn_cari.grid(row=0, column=4, padx=(15, 0))

    frm_input.columnconfigure(1, weight=1)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview",
                    background=PANEL, foreground=HITAM,
                    rowheight=26, fieldbackground=PANEL,
                    font=FONT_NORMAL)
    style.configure("Treeview.Heading",
                    background=ABU_MUDA, foreground=HITAM,
                    font=FONT_LABEL)
    style.map("Treeview", background=[("selected", "#9E3732")])

    frm_panel = tk.Frame(root, bg=PANEL)
    frm_panel.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    frm_info = tk.Frame(frm_panel, bg=PANEL, padx=12, pady=6)
    frm_info.pack(fill="x")
    lbl_info = tk.Label(frm_info,
                        text="Masukkan kata kunci lalu tekan Cari.",
                        font=FONT_KECIL, bg=PANEL, fg=ABU)
    lbl_info.pack(side="left")

    kolom_hasil = ("Peringkat", "ID Dok", "Skor Cosine", "Sumber", "Cuplikan Teks")
    tree_hasil  = ttk.Treeview(frm_panel, columns=kolom_hasil,
                                show="headings", selectmode="browse")

    for col, lebar in zip(kolom_hasil, [72, 68, 100, 130, 580]):
        tree_hasil.heading(col, text=col)
        tree_hasil.column(col, width=lebar, minwidth=40,
                          anchor="center" if lebar <= 130 else "w")

    sb_y = ttk.Scrollbar(frm_panel, orient="vertical", command=tree_hasil.yview)
    tree_hasil.configure(yscrollcommand=sb_y.set)
    tree_hasil.pack(side="left", fill="both", expand=True)
    sb_y.pack(side="right", fill="y")

    tree_hasil.tag_configure("ganjil", background="#EFF6FF")
    tree_hasil.tag_configure("genap",  background=PANEL)

    def lakukan_pencarian(event=None):
        query = var_query.get().strip()
        if not query:
            messagebox.showwarning("Perhatian", "Masukkan kata kunci terlebih dahulu.")
            return

        k     = var_topk.get()
        hasil = cari_indobert(query, embeddings_dokumen, tokenizer, model, device, top_k=k)

        for row in tree_hasil.get_children():
            tree_hasil.delete(row)

        if not hasil:
            lbl_info.config(
                text=f'Tidak ada dokumen relevan untuk "{query}".',
                fg=MERAH
            )
            return

        lbl_info.config(
            text=(f'Query: "{query}"  |  {len(hasil)} dokumen ditemukan  '
                  f'|  Skor tertinggi: {hasil[0][1]:.4f}'),
            fg=BIRU
        )

        for rank, (idx, skor) in enumerate(hasil, 1):
            baris   = df.iloc[idx]
            sumber  = LABEL_SUMBER.get(baris.get('sumber'), '?')
            preview = str(baris[kolom_asli])[:150].replace('\n', ' ')
            tag     = "ganjil" if rank % 2 else "genap"
            tree_hasil.insert("", "end", iid=str(rank), tags=(tag,),
                              values=(rank, baris['id_dok'],
                                      f"{skor:.4f}", sumber, preview))

    btn_cari.config(command=lakukan_pencarian)
    entry.bind("<Return>", lakukan_pencarian)
    entry.focus()

    root.mainloop()


# PROGRAM UTAMA

def evaluasi_terminal_indobert(df, embeddings_dokumen, tokenizer, model, device):
    import math
    queries = [
        "Penolakan vaksinasi oleh masyarakat dan kelompok antivaksin",
        "Dampak dan efektivitas vaksinasi terhadap penurunan kasus COVID-19",
        "Efek samping vaksin dan KIPI (Kejadian Ikutan Pasca Imunisasi)",
        "Diplomasi vaksin Indonesia di tingkat internasional",
        "Imunisasi anak dan vaksin campak untuk mencegah penyakit menular"
    ]
    ground_truths = [
        [8, 23, 29, 63, 69],
        [12, 27, 36, 37, 50],
        [3, 17, 40, 54, 70],
        [25, 32, 41, 42, 45],
        [52, 59, 60, 64, 74]
    ]
    
    print("\n")
    print("EVALUASI 5 GROUND TRUTH QUERY (INDOBERT)".center(60, " "))
    total_ap = 0; total_rr = 0; total_ndcg = 0
    
    for i, q in enumerate(queries):
        hasil = cari_indobert(q, embeddings_dokumen, tokenizer, model, device, top_k=5)
        doc_ids = [df.iloc[idx]['id_dok'] for idx, skor in hasil]
        gt = ground_truths[i]
        
        ap = sum([sum([1 for x in doc_ids[:k] if x in gt])/k for k, did in enumerate(doc_ids, 1) if did in gt]) / len(gt)
        rr = next((1.0/k for k, did in enumerate(doc_ids, 1) if did in gt), 0.0)
        dcg = sum([1.0/math.log2(k+1) for k, did in enumerate(doc_ids, 1) if did in gt])
        ndcg = dcg / 2.9485
        
        total_ap += ap; total_rr += rr; total_ndcg += ndcg
        
        print(f"\nQ{i+1}: {q}")
        print(f"Top-5 Dokumen: {doc_ids}")
        print(f"Metrics -> AP: {ap:.4f} | RR: {rr:.4f} | NDCG: {ndcg:.4f}")
        
    print("\n")
    print(f"RATA-RATA -> MAP: {total_ap/5:.4f} | MRR: {total_rr/5:.4f} | NDCG: {total_ndcg/5:.4f}")
    print("\n")
    print("Membuka Antarmuka Pengguna (UI)...")

def main():
    print("\n Load dataset")
    df, kolom_asli = muat_dataset(FILEPATH)

    print(f"\n '{MODEL_NAME}'...")
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model     = model.to(device)
    model.eval()
    print(f"      Model siap di {device}.")

    print(f"\n Load {len(df)} dokumen")
    teks_list          = df[kolom_asli].astype(str).tolist()
    embeddings_dokumen = ekstrak_embeddings(teks_list, tokenizer, model, device)
    print(f"      Selesai {embeddings_dokumen.shape}")

    evaluasi_terminal_indobert(df, embeddings_dokumen, tokenizer, model, device)

    jalankan_ui(df, kolom_asli, embeddings_dokumen, tokenizer, model, device)


if __name__ == '__main__':
    main()