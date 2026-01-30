package handlers

import (
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

var telemetrySchemas = map[string][]string {
	"PacketInfo": { "PacketID", "LapID", "PacketDatetime" },
	"DriverInputs": { "PacketID", "Gas", "Brake", "Steer", "Clutch", "Handbrake", "Gear" },
	"CarState": { "PacketID", "Fuel", "SpeedMPH", "RPM", "EngagedGear", "TurboBoost", "Weight", "WorldPositionX", "WorldPositionY", "WorldPositionZ", "AngularVelocityX", "AngularVelocityY", "AngularVelocityZ", "VelocityX", "VelocityY", "VelocityZ", "AccelerationX", "AccelerationY", "AccelerationZ", "Aero_DragCoeffcient", "Aero_LiftCoefficientFront", "Aero_LiftCoefficientRear", "CarForwardVectorX", "CarForwardVectorY", "CarForwardVectorZ", "CarSideVectorX", "CarSideVectorY", "CarSideVectorZ" },
	"ACState": { "PacketID", "ResetCount", "CollidedWith", "HeadlightsActive", "Ping", "SteerTorque" },
	"TireInfo": { "PacketID", "FL_Camber", "FR_Camber", "RL_Camber", "RR_Camber", "FL_ToeIn", "FR_ToeIn", "RL_ToeIn", "RR_ToeIn", "FL_TyreRadius", "FR_TyreRadius", "RL_TyreRadius", "RR_TyreRadius", "FL_TyreWidth", "FR_TyreWidth", "RL_TyreWidth", "RR_TyreWidth", "FL_RimRadius", "FR_RimRadius", "RL_RimRadius", "RR_RimRadius" },
	"TireState": { "PacketID", "FL_TyreWear", "FR_TyreWear", "RL_TyreWear", "RR_TyreWear", "FL_TyreVirtualMPH", "FR_TyreVirtualMPH", "RL_TyreVirtualMPH", "RR_TyreVirtualMPH", "FL_TyreDirtyLevel", "FR_TyreDirtyLevel", "RL_TyreDirtyLevel", "RR_TyreDirtyLevel", "FL_Slip", "FR_Slip", "RL_Slip", "RR_Slip", "FL_SlipAngle", "FR_SlipAngle", "RL_SlipAngle", "RR_SlipAngle", "FL_SlipRatio", "FR_SlipRatio", "RL_SlipRatio", "RR_SlipRatio", "FL_NDSlip", "FR_NDSlip", "RL_NDSlip", "RR_NDSlip", "FL_Load", "FR_Load", "RL_Load", "RR_Load", "FL_CoreTemperature", "FR_CoreTemperature", "RL_CoreTemperature", "RR_CoreTemperature", "FL_TyreInsideTemperature", "FR_TyreInsideTemperature", "RL_TyreInsideTemperature", "RR_TyreInsideTemperature", "FL_TyreMiddleTemperature", "FR_TyreMiddleTemperature", "RL_TyreMiddleTemperature", "RR_TyreMiddleTemperature", "FL_TyreOutsideTemperature", "FR_TyreOutsideTemperature", "RL_TyreOutsideTemperature", "RR_TyreOutsideTemperature", "FL_TyreOptimumTemperature", "FR_TyreOptimumTemperature", "RL_TyreOptimumTemperature", "RR_TyreOptimumTemperature", "FL_TemperatureMultiplier", "FR_TemperatureMultiplier", "RL_TemperatureMultiplier", "RR_TemperatureMultiplier", "FL_StaticPressure", "FR_StaticPressure", "RL_StaticPressure", "RR_StaticPressure", "FL_DynamicPressure", "FR_DynamicPressure", "RL_DynamicPressure", "RR_DynamicPressure", "FL_SelfAligningTorque", "FR_SelfAligningTorque", "RL_SelfAligningTorque", "RR_SelfAligningTorque", "FL_TyreContactNormalX", "FL_TyreContactNormalY", "FL_TyreContactNormalZ", "FR_TyreContactNormalX", "FR_TyreContactNormalY", "FR_TyreContactNormalZ", "RL_TyreContactNormalX", "RL_TyreContactNormalY", "RL_TyreContactNormalZ", "RR_TyreContactNormalX", "RR_TyreContactNormalY", "RR_TyreContactNormalZ" },
}

var lapSectorSchemas = map[string][]string {
	"LapInfo": { "LapID", "LapTime", "SectorConfiguration", "TrackName", "TrackConfiguration", "DriverName", "CarName", "PacketIDtart", "PacketIDEnd" },
	"SectorInfo": { "LapID", "SectorID", "SectorTime", "SectorStartPosX", "SectorStartPosY", "SectorEndPosX", "SectorEndPosY" },
}

func SqliteQuery(w http.ResponseWriter, r *http.Request, db *sql.DB) {
	start := r.URL.Query().Get("start")
	end := r.URL.Query().Get("end")
	table := r.URL.Query().Get("table")

	if start == "" || end == "" || table == "" {
		http.Error(w, "Missing start, end or table", 512)
		return
	}

	var query = fmt.Sprintf("SELECT * FROM %s WHERE PacketID >= ? AND PacketID <= ?", table)
	rows, err := db.Query(query, start, end)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var results []map[string]interface{}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		scanArgs := make([]interface{}, len(columns))
		for i := range values {
			scanArgs[i] = &values[i]
		}

		err = rows.Scan(scanArgs...)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			val := values[i]

			if b, ok := val.([]byte); ok {
				row[col] = string(b)
			} else {
				row[col] = val
			}
		}
		results = append(results, row)

		if err = rows.Err(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if len(results) == 0 {
			http.Error(w, "No records found in the specified range", 513)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(results)
	}
}
func AppendTelemetryCSV(w http.ResponseWriter, r *http.Request, db *sql.DB) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var currentPacketID int
	err := db.QueryRow("SELECT MAX(PacketID) AS CURRENTPACKETID FROM PacketInfo;").Scan(&currentPacketID)
	if err != nil {
		fmt.Printf("\n%s", err.Error())
		currentPacketID = 0
	}
	fmt.Printf("\nCurrent PacketID: %d", currentPacketID)

	csvReader := csv.NewReader(r.Body) // Read from request body
	defer r.Body.Close()               // Close request body after reading
	csvReader.FieldsPerRecord = -1
	csvReader.TrimLeadingSpace = true

	records, err := csvReader.ReadAll()
	if err != nil {
		http.Error(w, "Failed to read CSV data from request body", http.StatusBadRequest)
		return
	}

	if len(records) <= 1 {
		http.Error(w, "CSV data must have header and at least one data row", http.StatusBadRequest)
		return
	}

	tx, err := db.Begin()
	if err != nil {
		http.Error(w, "Failed to begin transaction", http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	insertedRowCount := 0
	headerRow := records[0]
	for i, record := range records[1:] {
		currentPacketID++
		insertedRowCount++
		
		if len(record) != len(headerRow) {
			http.Error(w, fmt.Sprintf("Row %d column count does not match header", i+2), http.StatusBadRequest)
			return
		}

		for tableName, schema := range telemetrySchemas {
			tableHeaders := make([]string, len(schema))
			tableValues := make([]string, len(schema))

			tableIndex := 0
			
			for colIdx, header := range headerRow {
				for _, schemaCol := range schema {
					if strings.EqualFold(header, schemaCol) {
						value := record[colIdx]
						if header == "PacketID" {
							tableHeaders[tableIndex] = header
							tableValues[tableIndex] = strconv.Itoa(currentPacketID)
							tableIndex++
						} else if value != "" { 
							tableHeaders[tableIndex] = header
							tableValues[tableIndex] = record[colIdx]
							tableIndex++
						}
						break 
					}
				}
			}
			if len(tableHeaders) == len(schema) {
				// insertQuery := "INSERT INTO " + tableName + " (" + strings.Join(tableHeaders, ", ") + ") VALUES (" + strings.Join(tableValues, ", ") + ");"
				
				// _, err := tx.Exec(insertQuery)
				// if err != nil {
				// 	fmt.Printf("Failed to insert into table '%s': %s", tableName, err.Error())
				// 	http.Error(w, fmt.Sprintf("Failed to insert into table '%s': %s", tableName, err.Error()), http.StatusInternalServerError)
				// 	return
				// }
				valuePlaceholders := make([]string, len(tableHeaders))
				for i := range tableHeaders {
					valuePlaceholders[i] = "?"
				}
				insertQuery := "INSERT INTO " + tableName + " (" + strings.Join(tableHeaders, ", ") + ") VALUES (" + strings.Join(valuePlaceholders, ", ") + ")"

				stmt, err := tx.Prepare(insertQuery)
				if err != nil {
					http.Error(w, fmt.Sprintf("Failed to prepare insert statement for table '%s': %s", tableName, err.Error()), http.StatusInternalServerError)
					return
				}
				defer stmt.Close()

				var args []interface{}
				for _, val := range tableValues {
					args = append(args, val)
				}

				_, err = stmt.Exec(args...)
				if err != nil {
					http.Error(w, fmt.Sprintf("Failed to execute insert statement for table '%s' in row %d: %s", tableName, i+2, err.Error()), http.StatusInternalServerError)
					return
				}
			}
		}
	}

	err = tx.Commit()
	if err != nil {
		http.Error(w, "Failed to commit transaction", http.StatusInternalServerError)
		return
	}
	
	responseMessage := fmt.Sprintf("\nCSV data appended successfully. Inserted %d new rows, re-assigned PacketIDs starting from %d.", insertedRowCount, currentPacketID+1)
	fmt.Println(responseMessage)
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(responseMessage))
}