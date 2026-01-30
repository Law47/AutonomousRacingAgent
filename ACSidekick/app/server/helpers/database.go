package helpers

import (
	"database/sql"
)

func SetupDatabaseSchema(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS PacketInfo (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			LapID INTEGER,
			PacketDatetime TEXT
		);

		CREATE TABLE IF NOT EXISTS LapInfo (
			LapID INTEGER UNIQUE PRIMARY KEY,
			LapTime REAL,
			SectorConfiguration TEXT,
			TrackName TEXT,
			TrackConfiguration TEXT,
			DriverName TEXT,
			CarName TEXT,
			PacketIDStart INTEGER,
			PacketIDEnd INTEGER
		);

		CREATE TABLE IF NOT EXISTS SectorInfo (
			LapID INTEGER,
			SectorID INTEGER,
			SectorTime REAL,
			SectorStartPosX REAL,
			SectorStartPosY REAL,
			SectorEndPosX REAL,
			SectorEndPosY REAL
		);

		CREATE TABLE IF NOT EXISTS DriverInputs (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			Gas REAL,
			Brake REAL,
			Steer REAL,
			Clutch REAL,
			Handbrake REAL,
			Gear INTEGER
		);

		CREATE TABLE IF NOT EXISTS CarState (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			Fuel REAL,
			SpeedMPH REAL,
			RPM REAL,
			EngagedGear INTEGER,
			TurboBoost REAL,
			Weight REAL,
			WorldPositionX REAL,
			WorldPositionY REAL,
			WorldPositionZ REAL,
			AngularVelocityX REAL,
			AngularVelocityY REAL,
			AngularVelocityZ REAL,
			VelocityX REAL,
			VelocityY REAL,
			VelocityZ REAL,
			AccelerationX REAL,
			AccelerationY REAL,
			AccelerationZ REAL,
			Aero_DragCoeffcient REAL,
			Aero_LiftCoefficientFront REAL,
			Aero_LiftCoefficientRear REAL,
			CarForwardVectorX REAL,
			CarForwardVectorY REAL,
			CarForwardVectorZ REAL,
			CarSideVectorX REAL,
			CarSideVectorY REAL,
			CarSideVectorZ REAL
		);

		CREATE TABLE IF NOT EXISTS ACState (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			ResetCount Integer,
			CollidedWith Integer,
			HeadlightsActive BOOL,
			Ping REAL,
			SteerTorque REAL
		);

		CREATE TABLE IF NOT EXISTS TireInfo (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			FL_Camber REAL,
			FR_Camber REAL,
			RL_Camber REAL,
			RR_Camber REAL,
			FL_ToeIn REAL,
			FR_ToeIn REAL,
			RL_ToeIn REAL,
			RR_ToeIn REAL,
			FL_TyreRadius REAL,
			FR_TyreRadius REAL,
			RL_TyreRadius REAL,
			RR_TyreRadius REAL,
			FL_TyreWidth REAL,
			FR_TyreWidth REAL,
			RL_TyreWidth REAL,
			RR_TyreWidth REAL,
			FL_RimRadius REAL,
			FR_RimRadius REAL,
			RL_RimRadius REAL,
			RR_RimRadius REAL
		);

		CREATE TABLE IF NOT EXISTS TireState (
			PacketID INTEGER UNIQUE PRIMARY KEY,
			FL_TyreWear REAL,
			FR_TyreWear REAL,
			RL_TyreWear REAL,
			RR_TyreWear REAL,
			FL_TyreVirtualMPH REAL,
			FR_TyreVirtualMPH REAL,
			RL_TyreVirtualMPH REAL,
			RR_TyreVirtualMPH REAL,
			FL_TyreDirtyLevel REAL,
			FR_TyreDirtyLevel REAL,
			RL_TyreDirtyLevel REAL,
			RR_TyreDirtyLevel REAL,
			FL_Slip REAL,
			FR_Slip REAL,
			RL_Slip REAL,
			RR_Slip REAL,
			FL_SlipAngle REAL,
			FR_SlipAngle REAL,
			RL_SlipAngle REAL,
			RR_SlipAngle REAL,
			FL_SlipRatio REAL,
			FR_SlipRatio REAL,
			RL_SlipRatio REAL,
			RR_SlipRatio REAL,
			FL_NDSlip REAL,
			FR_NDSlip REAL,
			RL_NDSlip REAL,
			RR_NDSlip REAL,
			FL_Load REAL,
			FR_Load REAL,
			RL_Load REAL,
			RR_Load REAL,
			FL_CoreTemperature REAL,
			FR_CoreTemperature REAL,
			RL_CoreTemperature REAL,
			RR_CoreTemperature REAL,
			FL_TyreInsideTemperature REAL,
			FR_TyreInsideTemperature REAL,
			RL_TyreInsideTemperature REAL,
			RR_TyreInsideTemperature REAL,
			FL_TyreMiddleTemperature REAL,
			FR_TyreMiddleTemperature REAL,
			RL_TyreMiddleTemperature REAL,
			RR_TyreMiddleTemperature REAL,
			FL_TyreOutsideTemperature REAL,
			FR_TyreOutsideTemperature REAL,
			RL_TyreOutsideTemperature REAL,
			RR_TyreOutsideTemperature REAL,
			FL_TyreOptimumTemperature REAL,
			FR_TyreOptimumTemperature REAL,
			RL_TyreOptimumTemperature REAL,
			RR_TyreOptimumTemperature REAL,
			FL_TemperatureMultiplier REAL,
			FR_TemperatureMultiplier REAL,
			RL_TemperatureMultiplier REAL,
			RR_TemperatureMultiplier REAL,
			FL_StaticPressure REAL,
			FR_StaticPressure REAL,
			RL_StaticPressure REAL,
			RR_StaticPressure REAL,
			FL_DynamicPressure REAL,
			FR_DynamicPressure REAL,
			RL_DynamicPressure REAL,
			RR_DynamicPressure REAL,
			FL_SelfAligningTorque REAL,
			FR_SelfAligningTorque REAL,
			RL_SelfAligningTorque REAL,
			RR_SelfAligningTorque REAL,
			FL_TyreContactNormalX REAL,
			FL_TyreContactNormalY REAL,
			FL_TyreContactNormalZ REAL,
			FR_TyreContactNormalX REAL,
			FR_TyreContactNormalY REAL,
			FR_TyreContactNormalZ REAL,
			RL_TyreContactNormalX REAL,
			RL_TyreContactNormalY REAL,
			RL_TyreContactNormalZ REAL,
			RR_TyreContactNormalX REAL,
			RR_TyreContactNormalY REAL,
			RR_TyreContactNormalZ REAL
		);
	`)
	return err
}

// ConvertSqlValue converts sql.Rows value types to Go types for JSON encoding.
// Specifically handles []byte to string conversion.
func ConvertSqlValue(val interface{}) interface{} {
	switch v := val.(type) {
	case []byte:
		return string(v)
	default:
		return v
	}
}
