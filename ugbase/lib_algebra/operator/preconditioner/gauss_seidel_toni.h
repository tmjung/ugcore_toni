/*
 * Copyright (c) 2026:  Toni
 *
 * This file is part of UG4.
 *
 * UG4 is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License version 3 (as published by the
 * Free Software Foundation) with the following additional attribution
 * requirements (according to LGPL/GPL v3 §7):
 *
 * (1) The following notice must be displayed in the Appropriate Legal Notices
 * of covered and combined works: "Based on UG4 (www.ug4.org/license)".
 *
 * (2) The following notice must be displayed at a prominent place in the
 * terminal output of covered works: "Based on UG4 (www.ug4.org/license)".
 *
 * (3) The following bibliography is recommended for citation and must be
 * preserved in all covered files:
 * "Reiter, S., Vogel, A., Heppner, I., Rupp, M., and Wittum, G. A massively
 *   parallel geometric multigrid solver on hierarchically distributed grids.
 *   Computing and visualization in science 16, 4 (2013), 151-164"
 * "Vogel, A., Reiter, S., Rupp, M., Nägel, A., and Wittum, G. UG4 -- a novel
 *   flexible software system for simulating pde based models on high performance
 *   computers. Computing and visualization in science 16, 4 (2013), 165-179"
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 */

#ifndef __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__GAUSS_SEIDEL_TONI__
#define __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__GAUSS_SEIDEL_TONI__

#include "lib_algebra/operator/preconditioner/gauss_seidel.h"

namespace ug{

/*
 * Planung Multi-Color Gauss-Seidel (MCGS), noch nicht implementiert:
 * Die Matrixzeilen/Blockzeilen werden nach ihrem Kopplungsgraphen gefaerbt.
 * Gekoppelte Indizes muessen verschiedene Farben haben. Farben werden
 * nacheinander bearbeitet; innerhalb einer Farbe sind Updates unabhaengig.
 * step() berechnet dabei c = B*d (Korrektur zum Defekt), nicht die Loesung u.
 * Details und Pruefplan: apps/Examples/multicolor_gauss_seidel_plan.md.
 *
 * TODO(MCGS-1): Dynamische Farbgruppen und Farbe je algebraischem Index
 * speichern. Graph aus der symmetrisierten OFF-DIAGONALEN Matrixstruktur
 * bilden: Verbindung i--j, wenn A(i,j) ODER A(j,i) strukturell vorhanden ist.
 * Konservativ auch gespeicherte Null-Eintraege beruecksichtigen; keine
 * numerische Schwellwert-Faerbung. Deterministisches Greedy reicht zum Start.
 * Nicht zwei Farben voraussetzen: Die Beispielgitter sind unstrukturiert.
 *
 * TODO(MCGS-5): OpenMP erst nach serieller Validierung hinzufuegen, mit einer
 * Barriere zwischen den Farben und privaten lokalen Rechenvariablen.
 * ACHTUNG: Auch SparseMatrix::const_row_iterator veraendert iIterators
 * (sparsematrix.h, add_iterator/remove_iterator); parallele Iterator-Nutzung
 * verursacht deshalb derzeit ein Datenrennen, auch im Release-Build.
 * Zuerst einen sicheren Lesezugriff planen, z.B. eine seriell vorbereitete
 * eigene Zeilenstruktur. Diagonalen ebenfalls vor der Thread-Region lesen.
 *
 * TODO(MCGS-6): MPI-Semantik separat festlegen. Die Basisklasse synchronisiert
 * c erst nach step(), nicht nach jeder Farbe. Lokale Faerbung ergibt daher
 * keinen globalen MCGS-Sweep. Eine erste serielle/OpenMP-Version muss mehrere
 * MPI-Prozesse explizit ablehnen oder einen dokumentierten lokalen Modus
 * anbieten; das geerbte supports_parallel()==true reicht als Zusage nicht.
 */
template <typename TAlgebra>
class GaussSeidelToni : public GaussSeidelBase<TAlgebra>
{
	typedef TAlgebra algebra_type;
	typedef typename TAlgebra::vector_type vector_type;
	typedef typename TAlgebra::matrix_type matrix_type;
	typedef GaussSeidelBase<TAlgebra> base_type;

public:
	virtual const char* name() const {return "Gauss-Seidel-Toni";}

	GaussSeidelToni() : base_type() {}

	GaussSeidelToni(const GaussSeidelToni<TAlgebra> &parent)
		: base_type(parent)
	{}

	// TODO(MCGS-2): preprocess(pOp) ueberschreiben und zuerst die Basisklasse
	// aufrufen (Quadratmatrix/Diagonale und MPI-Aufbereitung). Danach genau die
	// Matrix faerben, die base_type::step() nutzt: seriell *pOp, bei mehreren
	// Prozessen die aufbereitete m_A, ggf. mit Overlap und zusaetzlichen Zeilen.
	// Pro init() Farbgruppen neu aufbauen und pruefen; spaeter Cache-Gueltigkeit
	// fuer Struktur, Umnummerierung, Layout und Verfeinerung explizit verwalten.
	// Optional Diagonalinversen vorbereiten (siehe jacobi.h); Wertaenderungen
	// erfordern deren Aktualisierung, auch bei unveraenderter Matrixstruktur.
	// Noch keine zusaetzlichen Methoden oder Member: hier nur der Arbeitsplan.

	// TODO(MCGS-3): Neue Einstellungen im Kopierkonstruktor uebernehmen, aber
	// matrixabhaengige Farb-/Diagonal-Caches zunaechst ungueltig lassen. GMG
	// klont den Glaetter je Ebene und initialisiert ihn mit der Ebenenmatrix.
	virtual SmartPtr<ILinearIterator<vector_type> > clone()
	{
		return make_sp(new GaussSeidelToni<algebra_type>(*this));
	}

	virtual void step(const matrix_type &A, vector_type &c, const vector_type &d, const number relax)
	{
		// TODO(MCGS-4): Durch einen Farbsweep ersetzen: c am Anfang jedes Aufrufs
		// vollstaendig nullen; d unveraendert lassen. Fuer Farbe k gilt beim
		// ersten Vorwaertssweep (Skalarfall):
		// c[i] = relax/A(i,i) * (d[i] - Summe_{color[j]<k} A(i,j)*c[j]).
		// Entscheidend ist die Farbfolge, NICHT j<i. gs_step_LL laesst sich
		// deshalb nicht einfach in einer anders geordneten Zeilenschleife nutzen.
		// Blockalgebren mit MatMultAdd/InverseMatMult behandeln; die Bridge
		// instanziiert diese Template-Klasse fuer mehrere Algebra-Typen.
		// relax (SOR innerhalb des Sweeps) und set_damp (nachtraegliche Skalierung
		// in IPreconditioner::apply) getrennt halten; nicht doppelt anwenden.
		// Ein voller Durchlauf ueber ALLE Farben entspricht einem Glaetterschritt.
		// Spaeter Rueckwaerts-/symmetrischen Sweep gesondert definieren; bei
		// einem Ruecklauf desselben apply() die Korrektur nicht erneut nullen.
		// Bis zur Implementierung bleibt dies der Standard-Gauss-Seidel-Schritt.
		gs_step_LL(A, c, d, relax);
	}
};

} // end namespace ug

#endif // __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__GAUSS_SEIDEL_TONI__
