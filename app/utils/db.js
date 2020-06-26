'use strict';
import Sequelize from 'sequelize';

const db = new Sequelize(
  process.env.DB_NAME,
  process.env.DB_USERNAME,
  process.env.DB_PASSWORD,
  {
    host: process.env.DB_HOST,
    dialect: 'postgres',
    dialectOptions: {
      ssl: { rejectUnauthorized: false }
    }
  }
);

export default db;
